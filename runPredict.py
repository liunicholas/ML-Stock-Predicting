import os

fname = "terminalOutput"

def main():
    #overwrites files
    os.system(f"python predict.py | tee ./info/terminal/{fname}.txt")
    # execfile("python myTF.py |& tee terminalOutput.txt")

main()
