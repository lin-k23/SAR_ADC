def select(mode):
    dict = {
        "Single-Channel SAR": "sar",
        "3xTI-SAR": "tisar",
        "2-stage Pipe-Sar": "pipesar2s",
        "1st-order NS-SAR": "nssar1o1c",
        "1st-order NS-SAR(with chopping)": "nssar1o1ccp",
        "3-stage Pipe-Sar": "pipesar3shp",
    }
    return dict[mode]


# test for select
if __name__ == "__main__":
    print(select("Single-Channel SAR"))
    print(select("3xTI-SAR"))
    print(select("2-stage Pipe-Sar"))
    print(select("1st-order NS-SAR"))
    print(select("1st-order NS-SAR(with chopping)"))
    print(select("3-stage Pipe-Sar"))
