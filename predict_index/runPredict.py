from os import system
from shutil import copyfile

from parameters import *

terminalOutputPath = "./info/terminal/terminalOutput.txt"

def main():
    #overwrites files
    system(f"python ./predictIndex.py | tee {terminalOutputPath}")
    # execfile("python myTF.py |& tee terminalOutput.txt")

    #auto save terminal output when running on remote machine
    if remoteMachine:
        remoteSavedPath = f"{savedModelsPath}/{daysBefore}_{daysAhead}_{remoteVersionName}"
        copyfile(terminalOutputPath, f"{remoteSavedPath}/terminalOutput.txt")

main()
