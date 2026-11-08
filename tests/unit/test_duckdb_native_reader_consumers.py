"""DuckDB native readers preserve values, previews and connection ownership."""
import pandas as pd
import pytest

from buildml.core.errors import IngestError
from buildml.core.types import EngineName
from buildml.ingest.native_load import load_native_path


@pytest.mark.parametrize('format_name',['csv','tsv','parquet','arrow','arrow_stream'])
@pytest.mark.parametrize('limit',[None,0,3])
def test_real_native_reader_values_and_disclosures(tmp_path,format_name,limit):
    pytest.importorskip('duckdb')
    pa=pytest.importorskip('pyarrow')
    frame=pd.DataFrame({'row':[5,1,9,3,8],'text':['pear',None,"O'Reilly",'comma,tab\tquote"','end']})
    path=tmp_path/"O'Reilly.data"
    if format_name in {'csv','tsv'}:
        frame.to_csv(path,index=False,sep='\t' if format_name=='tsv' else ',')
    elif format_name=='parquet':
        frame.to_parquet(path,index=False)
    else:
        table=pa.Table.from_pandas(frame,preserve_index=False)
        writer=pa.ipc.new_stream if format_name=='arrow_stream' else pa.ipc.new_file
        with pa.OSFile(str(path),'wb') as sink, writer(sink,table.schema) as output:
            output.write_table(table)
    handle,schema,details=load_native_path(path,engine=EngineName.DUCKDB,format_name='arrow' if format_name=='arrow_stream' else format_name,nrows=limit,lazy=True)
    try:
        actual=handle.relation.df()
        expected=frame if limit is None else frame.head(limit)
        pd.testing.assert_frame_equal(actual.reset_index(drop=True),expected.reset_index(drop=True),check_dtype=False)
        assert [x.name for x in schema.fields] == list(frame.columns)
        assert details['n_rows'] == len(expected)
        assert details['lazy_handle'] is True
        assert details['lazy_scan'] is (not format_name.startswith('arrow'))
        assert details['pandas_first'] is False
    finally:
        handle.connection.close()


@pytest.mark.parametrize('format_name',['csv','parquet','arrow','unknown'])
def test_failed_reader_closes_owned_connection(tmp_path,monkeypatch,format_name):
    duckdb=pytest.importorskip('duckdb')
    real_connect=duckdb.connect
    connections=[]
    def connect(*args,**kwargs):
        connection=real_connect(*args,**kwargs)
        connections.append(connection)
        return connection
    monkeypatch.setattr(duckdb,'connect',connect)
    path=tmp_path/'broken.data'
    if format_name != 'csv':
        path.write_bytes(b'not a valid columnar file')
    try:
        with pytest.raises(IngestError):
            load_native_path(path,engine=EngineName.DUCKDB,format_name=format_name)
        assert connections
        with pytest.raises(duckdb.ConnectionException,match='closed'):
            connections[-1].execute('select 1')
    finally:
        for connection in connections:
            connection.close()
