import fire

def train(
    data: str,
    output: str
):
    
    print('Starting...')
    print(data)
    print(output)

if __name__ == "__main__":
    fire.Fire(train)
