

def quantify(x, width=16, frac=12):
    result = int(x * (2**frac)) & (2**width-1)
    print(f'raw: {x}, quantified value: {result}, hex is 0x{hex(result)[2:]}')
    return result
    


def dequantify(x, width=16, frac=12):
    if x > 1<<(width-1):
        print('minus')
        result = -((~(x-1))&(2**width-1)) / (2**frac)
    else:
        result = x / (2**frac)
    print(result)


