import numpy as np
from rlwe import RLWE,Rq
from Net import *
import utils
#这个函数每个参数都还有可调整的空间
def RLWE_INIT(Scale,num_clients):
    n = 1<<10
    #理论上n定了就与model无关了，但是可以考虑一个动态的n，在某个范围内尽可能充分利用空间，而且需要知道参数被分成了多少组
    t = utils.next_prime(num_clients*(10**Scale)*5)
    q = utils.next_prime(t*20)
    #q = 100_000_000_003 
    #t = 200_000_001
    std = 3
    rlwe = RLWE(n,q,t,std)
    #print("IN RLWE_INIT ",n,q,t)
    return q,t,std,rlwe


def Test_err(num_clients):
    Scale = 6 #参数直接放大的倍数
    n = 1<<10
    q,t,std,rlwe = RLWE_INIT(Scale,num_clients)
    '''n = 8 #(2^3)
    t = 29
    q = 1019 
    std = 3
    rlwe = RLWE(n,q,t,std)'''
    rlwe.generate_vector_a()

    #m1 = Rq([-9435275, 9030923, 9230815, 6849630, -2557183, -1291996, 2815141, 27342],t)
    #m2 = Rq([-4105284, 8620511, -285281, 4626085, 625026, 4053945, 467085, -2194348],t)
    #m3 = Rq([-1,-1,-1,-1,1,1,1,1],t)
    #print("t={},q={}".format(t,q))
    Max = t-1000000
    Mid = Max//2
    m = [Rq(np.random.randint(Max,size = n)-Mid,t) for _ in range(num_clients)]
    plain_sum  = Rq([0],t)
    for mm in m:
        plain_sum += mm
    s = []
    p = []
    sumb = Rq([0],q)
    for i in range(num_clients):
        ss,pp = rlwe.generate_keys()
        s.append(ss)
        p.append(pp)
        sumb += pp[0]
    
    #print("s1",s1)
    #print("p1",p1)
    c = []
    csum0,csum1 = Rq([0],q),Rq([0],q)
    for i in range(num_clients):
        c.append(rlwe.encrypt(m[i],(sumb,p[i][1])))
        csum0 += c[i][0]
        csum1 += c[i][1]
    #print('c1',c1)
    #print('csum0',csum0[:8])
    #print('csum1',csum1[:8])
    err = [Rq(np.round(std*np.random.randn(n)),q) for _ in range(num_clients)]
    dsum = Rq([0],q)
    for i in range(num_clients):
        dsum += rlwe.decrypt(csum1,s[i],err[i])
    
    #d2 = rlwe.decrypt(csum1,s2,err2)
    #d3 = rlwe.decrypt(csum1,s3,err3)
    #print('d1',d1[:8])
    #print('d2',d2[:8])
    #dsum = d1+d2+d3
    #print('dsum',dsum[:8])
    m_ = Rq((dsum+csum0).poly.coeffs,t)
    plain_sum = plain_sum.poly_to_list()
    m_ = m_.poly_to_list()
    #print("plain_sum",plain_sum[:8])
    #print("sum",m_[:8])
    ret = 0
    for i in range(n):
        ret += abs(m_[i]-plain_sum[i])*1./abs(plain_sum[i])
    #print("ret",ret)
    if ret/n>1:
        print("plain_sum",plain_sum[:8])
        print("sum",m_[:8])
    return ret/n

if __name__ == '__main__':
    print("In test-rlwe")
    #model = Binary_CF()
    client_numlst = [3,5,7,10,15,20]
    for client_num in client_numlst:
        toterr = 0.0
        for i in range(1000):
            toterr += Test_err(client_num)
        print("avgerr for {} clients is {}".format(client_num,toterr/1000))



