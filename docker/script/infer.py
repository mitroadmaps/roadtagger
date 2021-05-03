# python 2
import requests
import json 
import argparse
import math  
import base64


def parseArgument():
    parser = argparse.ArgumentParser()

    parser.add_argument('-image', action='store', dest='inputimage', type=str,
                        help='input image file', required =True)
    
    parser.add_argument('-graph', action='store', dest='inputgraph', type=str,
                        help='input graph file', required =True)
    
    parser.add_argument('-gsd', action='store', dest='gsd', type=float,
                        help='ground sample distance', required =False, default= 1)
    
    parser.add_argument('-output', action='store', dest='output', type=str,
                        help='output graph (json format)', required =False, default="out.json")
    

    return parser.parse_args()

if __name__ == "__main__":
    args = parseArgument()

    img_bin = open(args.inputimage,"rb")
    img_base64 = base64.encodestring(img_bin.read())
    print(type(img_base64))
    msg = {}
    msg["inputtype"] = "base64"
    msg["imagebase64"] = img_base64
    msg["imagetype"] = args.inputimage.split(".")[-1] 
    msg["gsd"] = args.gsd
    msg["graph"] = json.load(open(args.inputgraph))

    
    print("running... check docker log for details")
    url = "http://localhost:8010"
    x = requests.post(url, data = json.dumps(msg))
    result = json.loads(x.text) 
    if result["success"] == 'false':
        print("unknown error, check docker log for details")
        exit()

    json.dump(result["result"], open(args.output, "w"), indent=2)
    



