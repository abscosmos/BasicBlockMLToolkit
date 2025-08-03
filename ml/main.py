from bb_toolkit import TraceData

def main():
    trace = TraceData.from_binary_file("../traces/ls_trace-688eab66.trace")
    print(len(trace.blocks))


if __name__ == "__main__":
    main()
