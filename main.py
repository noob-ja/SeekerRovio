from lib import rovio
import numpy as np
from movement import movementAlgorithm

class rovioControl:
    def __init__(self,url, username=None, password=None, port = 80):
        self.rovio = rovio.Rovio(url,username=username,password=password,port = port)
        self.last = None
        self.key = 0

    def main(self):
        move = movementAlgorithm(self.rovio)
        move.start()

if __name__ == "__main__":
    url = '192.168.43.134'
    app = rovioControl(url)
    app.main()
