from os import system
from shutil import copy2

from parameters import *

terminalOutputPath = "./info/terminal/terminalOutput.txt"

def main():
    #overwrites files
    system(f"python ./predictIndex.py | tee {terminalOutputPath}")
    # execfile("python myTF.py |& tee terminalOutput.txt")
    if remoteMachine:
        remoteSavedPath = f"{savedModelsPath}/{remoteVersionName}"
        copy2(terminalOutputPath, remoteSavedPath)

main()
