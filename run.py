# CLI to price and compare methods
# Usage : python -m cli.run --method bsm --S 100 --K 100 --sigma 0.2 --T 0.5

import argparse
from scripts import binomial_price, bsm_price, fd_price_cn, mc_price

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['bsm', 'binomial', 'fd', 'mc'], default= 'bsm')
    p.add_argument('--S', type= float, default= 100.0)
    p.add_argument('--K', type= float, default= 100.0)
    p.add_argument('--r', type= float, default= 0.01)
    p.add_argument('--q', type= float, default= 0.0)
    p.add_argument('--sigma', type= float, default= 0.2)
    p.add_argument('--T', type= float, default= 0.5)
    p.add_argument('--steps', type= int, default= 200)
    p.add_argument('--npaths', type= int, default= 50000)
    args = p.parse_args()

    if args.method == 'bsm':
        price = bsm_price(args.S, args.K, args.r, args.q, args.sigma, args.T, option_type= 'call')
    
    elif args.method == 'binomial':
        price = binomial_price(args.S, args.K, args.r, args.q, args.sigma, args.T, option_type= 'call')
    
    elif args.method == 'fd':
        price = fd_price_cn(args.S, args.K, args.r, args.q, args.sigma, args.T, option_type= 'call')
    
    else :
        price = mc_price(args.S, args.K, args.r, args.q, args.sigma, args.T, option_type= 'call')

if __name__ == '__main__' :
    main()
    