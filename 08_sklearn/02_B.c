/'''import math
def sieve(size):
    sieve = [True]*size
    sieve[0] = False
    sieve[1] = False

    for i in range(2,int(math.sqrt(size))+1):
        k = 2*i
        while k < size:
            sieve[k] = False
            k = k+i
    
    return sum(1 for x in sieve if x)

print(sieve(10))'''/
void main(){
    int a[5];
    print("1.%d",sizeof(a));
}