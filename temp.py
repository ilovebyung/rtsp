def f():
    image = 1
    result = 0
    return image, result

def g():
    image = None
    result = None

    def check():
        nonlocal image
        nonlocal result
        try:
            image, result = f()
            return image, result
        except:
            print('an exception has occurred')

r=g()
print(r)
