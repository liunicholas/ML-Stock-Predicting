import os

terminalOutputPath = "./terminalOutput.txt"

def main():
    #overwrites files
    os.system(f"python hi.py | tee {terminalOutputPath}")
    # execfile("python myTF.py |& tee terminalOutput.txt")

main()
