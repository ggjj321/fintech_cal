#############################################################
# Problem 0: Find base point
def GetCurveParameters():
    # Certicom secp256-k1
    # Hints: https://en.bitcoin.it/wiki/Secp256k1
    _p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    _a = 0x0000000000000000000000000000000000000000000000000000000000000000
    _b = 0x0000000000000000000000000000000000000000000000000000000000000007
    _Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
    _Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
    _Gz = 0x0000000000000000000000000000000000000000000000000000000000000001
    _n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    _h = 0x01
    return _p, _a, _b, _Gx, _Gy, _Gz, _n, _h


#############################################################
# Problem 1: Evaluate 4G
def compute4G(G, callback_get_INFINITY):
    """Compute 4G"""

    """ Your code here """
    result = callback_get_INFINITY() 
    for _ in range(4):
        result += G
        
    return result 


#############################################################
# Problem 2: Evaluate 5G
def compute5G(G, callback_get_INFINITY):
    """Compute 5G"""

    """ Your code here """
    result = callback_get_INFINITY() 
    for _ in range(5):
        result += G
        
    return result 


#############################################################
# Problem 3: Evaluate dG
# Problem 4: Double-and-Add algorithm
def double_and_add(n, point, callback_get_INFINITY):
    """Calculate n * point using the Double-and-Add algorithm."""

    """ Your code here """
    if n == 0:
        return callback_get_INFINITY(), 0, 0

    if n == 1:
        return point, 0, 0
    
    result = callback_get_INFINITY()
    num_doubles = 0 
    num_additions = 0  

    binary_n = bin(n)[3:] 
    result = result + point

    for bit in binary_n:
        result = result.double()
        num_doubles += 1

        if bit == '1':
            result = result + point
            num_additions += 1

    return result, num_doubles, num_additions

#############################################################
# Problem 5: Optimized Double-and-Add algorithm
def optimized_double_and_add(n, point, callback_get_INFINITY):
    """Optimized Double-and-Add algorithm that simplifies sequences of consecutive 1's."""

    """ Your code here """
    result = callback_get_INFINITY()
    num_doubles = 0
    num_additions = 0
    
    def optimized_alg(binary_num, result, num_doubles, num_additions):
        if binary_num[0] == "1" and all(char == '0' for char in binary_num[1:]):
            zero_count = binary_num[1:].count('0')
            
            for _ in range(zero_count):
                result = result.double()
                num_doubles += 1
            return binary_num, result, num_doubles, num_additions
        
        if binary_num[-1] == "0":
            binary_num = binary_num[:-1]
            binary_num, result, num_doubles, num_additions = optimized_alg(binary_num, result, num_doubles, num_additions)
            result = result.double()
            num_doubles += 1
            return binary_num, result, num_doubles, num_additions
        
        if binary_num[-1] == "1":
            converted_int = int(binary_num, 2)
            converted_int += 1
            converted_binarty = bin(converted_int)[2:-1]
            binary_num, result, num_doubles, num_additions = optimized_alg(converted_binarty, result, num_doubles, num_additions)
            result = result.double()
            result = result + (-point)
            num_doubles += 1
            num_additions += 1
            return binary_num, result, num_doubles, num_additions
    
    binary_num = bin(n)[2:] 
    result = result + point
    
    binary_num, result, re_num_doubles, re_num_additions = optimized_alg(binary_num, result, num_doubles, num_additions)
    
    return binary_num, result, re_num_doubles, re_num_additions


#############################################################
# Problem 6: Sign a Bitcoin transaction with a random k and private key d
def sign_transaction(private_key, hashID, callback_getG, callback_get_n, callback_randint):
    """Sign a bitcoin transaction using the private key."""

    """ Your code here """
    G = callback_getG()
    n = callback_get_n()
    signature = callback_randint()

    return signature


##############################################################
# Step 7: Verify the digital signature with the public key Q
def verify_signature(public_key, hashID, signature, callback_getG, callback_get_n, callback_get_INFINITY):
    """Verify the digital signature."""

    """ Your code here """
    G = callback_getG()
    n = callback_get_n()
    infinity_point = callback_get_INFINITY()
    is_valid_signature = TRUE if callback_get_n() > 0 else FALSE

    return is_valid_signature

