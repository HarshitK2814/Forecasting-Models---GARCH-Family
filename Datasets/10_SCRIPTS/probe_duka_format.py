import urllib.request, ssl, certifi, lzma, struct
from datetime import datetime, timedelta
CTX=ssl.create_default_context(cafile=certifi.where())
def raw(ins,y,m,d,kind="BID_candles_min_1"):
    # NOTE: Dukascopy months are ZERO-INDEXED in the URL
    u=f"https://datafeed.dukascopy.com/datafeed/{ins}/{y:04d}/{m-1:02d}/{d:02d}/{kind}.bi5"
    req=urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0'})
    try:
        b=urllib.request.urlopen(req,timeout=30,context=CTX).read()
        return u,b
    except Exception as e:
        return u,f"ERR {type(e).__name__} {e}"

for (ins,y,m,d) in [("USA500IDXUSD",2020,3,16),("USA500IDXUSD",2015,6,10)]:
    u,b=raw(ins,y,m,d)
    if isinstance(b,str): print(u,"->",b); continue
    print(f"{u}\n  bytes={len(b)}")
    try:
        dec=lzma.LZMADecompressor().decompress(b)
    except Exception as e:
        print("  lzma fail",e); continue
    print(f"  decompressed={len(dec)}  /24={len(dec)/24}  /20={len(dec)/20}")
    n=len(dec)//24
    print("  first 3 records assuming >IfffffI(24B) = time,open,close,low,high,vol:")
    for i in range(3):
        t,o,c,lo,hi,v=struct.unpack('>Ifffff',dec[i*24:(i+1)*24])
        print(f"    t={t:>7} ({timedelta(seconds=t)})  O={o:.2f} C={c:.2f} L={lo:.2f} H={hi:.2f} V={v:.3f}")
    print("  last 2:")
    for i in (n-2,n-1):
        t,o,c,lo,hi,v=struct.unpack('>Ifffff',dec[i*24:(i+1)*24])
        print(f"    t={t:>7} ({timedelta(seconds=t)})  O={o:.2f} C={c:.2f} L={lo:.2f} H={hi:.2f} V={v:.3f}")
