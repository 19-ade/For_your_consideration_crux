import numpy as np
import string as str
import re

f = open("/home/anwesan/Downloads/WhatsApp Chat with Skwad.txt", "r")


def manipulations(fname_path):
    num_line = counter_lines_execution(fname_path)

    f = open(fname_path, "r")
    s = []

    for i in range(num_line):
        p = f.readline()
        pattern = "{2}[/-]\d{2}[/-]\d{2}"
        n = p[:9]
        x = re.findall(r'(\d+/\d+/\d+)', n)
        if (x != None):
            p = p[21:]

        '''if p.startswith("\u200e"):
            p=p[30:]
        if (p.find("\u200e")>0):
            continue'''
        if (p.startswith("Satwik 134:")) or (p.startswith("atwik 134:")) or (p.startswith("Abhinav 133:")) or (
                p.startswith("irav:")) or (p.startswith("Nirav:")) or (p.startswith("Aditya:")) or (
        p.startswith("ditya:")) or (
                p.startswith("Anwesan 133:")) or (p.startswith("nwesan 133:")) or p.startswith("bhinav 133:"):
            if p.startswith("Satwik 134:"):
                p = p[12:]
            elif p.startswith("atwik 134:"):
                p = p[11:]
            elif p.startswith("bhinav 133:"):
                p = p[12:]
            elif p.startswith("Abhinav 133:"):
                p = p[13:]
            elif p.startswith("Nirav:"):
                p = p[7:]
            elif p.startswith("irav:"):
                p = p[6:]
            elif p.startswith("Aditya:"):
                p = p[8:]
            elif p.startswith("ditya:"):
                p = p[7:]
            elif p.startswith("Anwesan 133:"):
                p = p[13:]
            elif p.startswith("nwesan 133:"):
                p = p[12:]
        if p.endswith("\n"):
            p = p[:-1]
        if p.startswith('<Media omitted>'):
            continue
        if p == ".":
            continue
        if p == '':
            continue
        if (p.startswith("This message was deleted")):
            # print("f")
            continue
        if p.startswith("atwik 134:"):
            print("f")
            p=p[11:]
        y=p.replace("\u200d♂️","")
        s.append(y)
    #print(len(s))
    # s.remove("atwik 134: This message was deleted")
    #print(s)
    return s


def counter_lines_execution(fname):
    num_lines = 0
    with open(fname, 'r') as f:
        for line in f:
            num_lines += 1
    # print("Number of lines:")

    # print(num_lines)
    return num_lines


# manipulations(f)
s = manipulations("/home/anwesan/Downloads/WhatsApp Chat with Skwad.txt")
# print(s[11])
