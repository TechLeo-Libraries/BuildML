This fixture was created by the public Session.synthetic.save_bundle API using
archived commit 62a90adc9687e50985a14405f3b68abfedd82711, whose complete CI passed.
It contains a native Gaussian copula fitted on 80 rows of an object column
alternating "left" and "right", with random_state=3. The original frame had120
rows; rows80:100 and100:120 were held out. No holdout was fitted.

Producer runtime:Python3.12.8/NumPy2.4.6/Pandas3.0.5 on Windows. The generator
contains NumPy arrays and scalar strings; it has no fitted pandas frame. The
actual old class schema predates raw categorical vocabulary storage. This is a
trusted repository test artifact, loaded only with explicit trusted=True.

Regression:loading an older slotted ColumnSchemaSpec must initialize newly
introduced fitted metadata to its legacy default. Public equality, JSON
summaries, sampling and saving the loaded bundle must continue to work.
Sampling keeps the legacy fitted population and warns that refitting is needed
to guarantee scalar category identity. JSON metadata retains its v1 format.
