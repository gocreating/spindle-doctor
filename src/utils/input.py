import pandas as pd

def get_batch(df_chunks, chunk_size, batch_size):
    df_buffer = pd.DataFrame()
    batch_idx = 0

    for df_chunk in df_chunks:
        # append new chunk
        df_buffer = df_buffer.append(df_chunk)
        buffer_len = len(df_buffer)

        while buffer_len > batch_size:
            yield batch_idx, df_buffer.iloc[0:batch_size]
            # drop yielded batch
            df_buffer = df_buffer.iloc[batch_size:]
            buffer_len -= batch_size
            batch_idx += 1

    # yield the last batch
    yield batch_idx, df_buffer
