from bb_toolkit import TraceData
from ml.tokenizer import BasicBlockTokenizer

def main():
    trace = TraceData.from_binary_file("../traces/ls_trace-688eab66.trace")
    tokenizer = BasicBlockTokenizer()

    sequence = tokenizer.process_trace(trace)

    print(sequence[:20])

    for i in range(6,10):
        block = trace.blocks[trace.order[i]]

        tk = sequence[i]
        print(f"{tk}:\n{tokenizer.get_block(tk)}\n=>\n{block}\n")


if __name__ == "__main__":
    main()
