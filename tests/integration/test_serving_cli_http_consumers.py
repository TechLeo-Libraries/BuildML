"""The CLI launches a real model server, enforces auth and stops cleanly."""
import base64
import os
import socket
import threading
import time

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from buildml import Session
from buildml.serving.cli import main


@pytest.mark.parametrize('auth',['none','api-key','basic'])
def test_real_cli_prediction_auth_and_blocking_shutdown(tmp_path,monkeypatch,capsys,auth):
    uvicorn=pytest.importorskip('uvicorn')
    httpx=pytest.importorskip('httpx')
    for name in list(os.environ):
        if name.startswith('BUILDML_'):
            monkeypatch.delenv(name)
    frame=pd.DataFrame({'x':np.arange(40,dtype=float)})
    frame['y']=2*frame.x+3
    s=Session.ingest(frame).set_roles({'x':'feature','y':'target'}).split(test_size=.2,random_state=0)
    s.fit(LinearRegression(),task='regression')
    bundle=tmp_path/'model'
    s.save_pipeline(bundle)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        port=sock.getsockname()[1]
    servers=[]
    original=uvicorn.Server
    def observe(*args,**kwargs):
        server=original(*args,**kwargs)
        servers.append(server)
        return server
    monkeypatch.setattr(uvicorn,'Server',observe)
    argv=['--bundle',str(bundle),'--port',str(port),'--trusted','--no-docs']
    headers={}
    if auth=='api-key':
        argv+=['--api-key','cli-test-secret']
        headers={'X-API-Key':'cli-test-secret'}
    elif auth=='basic':
        argv+=['--basic-auth','tester:cli-test-secret']
        token=base64.b64encode(b'tester:cli-test-secret').decode()
        headers={'Authorization':'Basic '+token}
    returns=[]
    errors=[]
    def run():
        try:
            returns.append(main(argv))
        except BaseException as exc:
            errors.append(exc)
    thread=threading.Thread(target=run,daemon=True)
    thread.start()
    try:
        deadline=time.monotonic()+15
        with httpx.Client(base_url=f'http://127.0.0.1:{port}',timeout=1) as client:
            while not servers or not servers[0].started:
                assert thread.is_alive(),repr(errors)
                assert time.monotonic()<deadline,'CLI startup exceeded15s'
                time.sleep(.02)
            payload={'rows':[{'x':4.},{'x':-7.}]}
            if auth!='none':
                assert client.post('/predict',json=payload).status_code==401
            response=client.post('/predict',json=payload,headers=headers)
            assert response.status_code==200,response.text
            np.testing.assert_allclose(response.json()['predictions'],[11.,-11.],atol=1e-10)
            assert client.get('/docs',headers=headers).status_code==404
            assert client.get('/health',headers=headers).status_code==200
            assert thread.is_alive()  # main remains blocking until shutdown.
    finally:
        for server in servers:
            server.should_exit=True
        thread.join(timeout=10)
    assert not thread.is_alive()
    assert not errors,repr(errors)
    assert returns==[0]
    stderr=capsys.readouterr().err
    assert f'http://127.0.0.1:{port}' in stderr
    assert ('auth off' if auth=='none' else auth+' auth on') in stderr
    assert 'cli-test-secret' not in stderr
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1',port))!=0
