import os

def main():
    #overwrites files
    os.system("python predict.py | tee terminal/terminalOutput.txt")
    # execfile("python myTF.py |& tee terminalOutput.txt")

main()
