def convert(s: str, numRows: int) -> str:
    numCols = len(s) // numRows + 1
    arr = [["" for x in range(numCols)] for y in range(numRows)]

    print(arr)

convert("PAYPAL", 3)