import os

terminalOutputPath = "./info/terminal/terminalOutput.txt"

def main():
    #overwrites files
    os.system(f"python momentum.py | tee {terminalOutputPath}")

main()
