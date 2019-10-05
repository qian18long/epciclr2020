from multiprocessing import Process, Pipe
from .train_helpers import train


class TrainProxy(Process):
    def __init__(self, kwargs, conn):
        Process.__init__(self, daemon=False)

        self.kwargs = kwargs
        self.conn = conn

    def run(self):
        self.conn.send(train(**self.kwargs))


def proxy_train(kwargs_list):
    not_list = False
    if type(kwargs_list) != list:
        not_list = True
        kwargs_list = [kwargs_list]

    train_proxies = []
    conns = []
    for kwargs in kwargs_list:
        conn_pair = Pipe()
        train_proxies.append(TrainProxy(kwargs, conn_pair[0]))
        conns.append(conn_pair[1])

    for train_proxy in train_proxies:
        train_proxy.start()

    result = []
    for conn in conns:
        result.append(conn.recv())

    for train_proxy in train_proxies:
        train_proxy.join()

    return result[0] if not_list else result