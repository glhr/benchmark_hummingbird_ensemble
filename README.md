Note: I have tested this with Python 3.10, but will probably also work with older versions.

## To install the necessary python dependencies:

```
pip install https://github.com/glhr/hummingbird/archive/refs/heads/main.zip
```

## To run the benchmark:
```
python3 run.py
```

if there are any issues running the code, let me know :)

Expected output:
```
Running benchmark with CPU
...Warmup...
Single converter case - Throughput: 9074329.598076593 predictions/sec
Parallel converter case - Throughput: 9211758.952215407 predictions/sec
```
