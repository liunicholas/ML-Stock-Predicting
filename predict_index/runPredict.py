import os

terminalOutputPath = "./info/terminal/terminalOutput.txt"

def main():
    #overwrites files
    os.system(f"python ./predictIndex.py | tee {terminalOutputPath}")
    # execfile("python myTF.py |& tee terminalOutput.txt")

main()
