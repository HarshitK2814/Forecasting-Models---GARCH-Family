import urllib.request, ssl, certifi, lzma, struct
from datetime import timedelta
CTX=ssl.create_default_context(cafile=certifi.where())
def get(ins,y,m,d,kind="BID_candles_min_1"):
    u=f"https://datafeed.dukascopy.com/datafeed/{ins}/{y:04d}/{m-1:02d}/{d:02d}/{kind}.bi5"
    req=urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0'})
    b=urllib.request.urlopen(req,timeout=30,context=CTX).read()
    return lzma.LZMADecompressor().decompress(b)

dec=get("USA500IDXUSD",2020,3,16)
n=len(dec)//24
print("2020-03-16 USA500IDXUSD (COVID crash Monday). Parsing >IiiiiF :")
nz=[]
for i in range(n):
    t,o,c,lo,hi,v=struct.unpack('>Iiiiif',dec[i*24:(i+1)*24])
    if o or c: nz.append((t,o,c,lo,hi,v))
print(f"  non-zero minutes: {len(nz)} / {n}")
for t,o,c,lo,hi,v in nz[:4]+nz[len(nz)//2:len(nz)//2+3]+nz[-3:]:
    print(f"   {str(timedelta(seconds=t)):>9}  raw O={o} C={c} L={lo} H={hi} V={v:.3f}")
print("\n  /1000 scaling ->")
for t,o,c,lo,hi,v in nz[len(nz)//2:len(nz)//2+3]:
    print(f"   {str(timedelta(seconds=t)):>9}  O={o/1000:.3f} C={c/1000:.3f} L={lo/1000:.3f} H={hi/1000:.3f} V={v:.3f}")
print("\n(S&P 500 closed 2020-03-16 at ~2386)")
