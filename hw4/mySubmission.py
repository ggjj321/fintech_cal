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
# Problem 6: Sign a Bitcoin transaction with a random k and private key d
def sign_transaction(private_key, hashID, callback_getG, callback_get_n, callback_randint):
    """Sign a bitcoin transaction using the private key."""
    def mod_inverse(a, n):
        t, newt = 0, 1
        r, newr = n, a
        while newr != 0:
            quotient = r // newr
            r, newr = newr, r - quotient * newr
            t, newt = newt, t - quotient * newt
        if r > 1:
            raise Exception('a is not invertible')
        if t < 0:
            t = t + n
        return t
    
    def calculate_z(n, e):
        L_n = n.bit_length()
        binary_e = bin(e)[2:]  # 將 e 轉為二進位，去掉 '0b' 前綴
        if len(binary_e) > L_n:
            z_binary = binary_e[:L_n]
        else:
            z_binary = binary_e.zfill(L_n)
        return int(z_binary, 2)

    G = callback_getG()  
    n = callback_get_n()  
    e = int(hashID, 16)  
    z = calculate_z(n, e)  

    while True:
        k = callback_randint(1, n - 1)
        
        def double_and_add(k, G):
            result = None  # 無窮遠點
            current = G
            for bit in bin(k)[2:]:  # 二進制逐位處理
                if result is None:
                    result = current
                else:
                    result = result.double()
                    if bit == '1':
                        result = result + current
            return result
        
        point = double_and_add(k, G)
        x = point.x()

        r = x % n
        if r == 0:
            continue

        k_inv = mod_inverse(k, n)
        s = (k_inv * (z + r * private_key)) % n
        if s == 0:
            continue  
        
        return (r, s)


##############################################################
# Step 7: Verify the digital signature with the public key Q
def verify_signature(public_key, hashID, signature, callback_getG, callback_get_n, callback_get_INFINITY):
    """Verify the digital signature."""

    """ Your code here """
    G = callback_getG()
    n = callback_get_n()
    infinity_point = callback_get_INFINITY()
    r, s = signature
    
    if public_key == infinity_point:
        return False
    
    if not (1 <= r < n and 1 <= s < n):
        return False
    
    e = int(hashID, 16)
    
    w = pow(s, n-2, n) 
    
    u1 = (e * w) % n
    u2 = (r * w) % n
    
    point = u1 * G + u2 * public_key

    if point == infinity_point:
        return False

    x1 = point.x() % n
    
    return x1 == r