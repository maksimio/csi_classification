// COMPILATION: gcc -shared -fPIC -o libextract_csi.so extract_csi.c

#define BITS_PER_BYTE 8
#define BITS_PER_SYMBOL 10

int signbit_convert(int data, int maxbit)
{
    if (data & (1 << (maxbit - 1)))
        data -= (1 << maxbit);
    return data;
}

void read_csi(unsigned char *local_h, int *re0, int *re1, int *re2, int *re3, int *im0, int *im1, int *im2, int *im3)
{
    unsigned int nr = 2, nc = 2, num_tones = 56;
    int real, imag;
    int bits_left = 16;

    unsigned int bitmask = (1 << BITS_PER_SYMBOL) - 1;
    unsigned int idx = 0;
    unsigned int h_data = local_h[idx++];
    h_data += (local_h[idx++] << BITS_PER_BYTE);
    unsigned int current_data = h_data & ((1 << 16) - 1);

    for (int k = 0; k < num_tones; k++)
        for (int nc_idx = 0; nc_idx < nc; nc_idx++)
            for (int nr_idx = 0; nr_idx < nr; nr_idx++)
            {
                if ((bits_left - BITS_PER_SYMBOL) < 0)
                {
                    h_data = local_h[idx++];
                    h_data += (local_h[idx++] << BITS_PER_BYTE);
                    current_data += h_data << bits_left;
                    bits_left += 16;
                }

                imag = current_data & bitmask;

                bits_left -= BITS_PER_SYMBOL;
                current_data = current_data >> BITS_PER_SYMBOL;

                if ((bits_left - BITS_PER_SYMBOL) < 0)
                {
                    h_data = local_h[idx++];
                    h_data += (local_h[idx++] << BITS_PER_BYTE);
                    current_data += h_data << bits_left;
                    bits_left += 16;
                }

                real = current_data & bitmask;

                switch (nr_idx + nc_idx * 2)
                {
                case 0:
                    im0[k] = signbit_convert(imag, BITS_PER_SYMBOL);
                    re0[k] = signbit_convert(real, BITS_PER_SYMBOL);
                    break;
                case 1:
                    im1[k] = signbit_convert(imag, BITS_PER_SYMBOL);
                    re1[k] = signbit_convert(real, BITS_PER_SYMBOL);
                    break;
                case 2:
                    im2[k] = signbit_convert(imag, BITS_PER_SYMBOL);
                    re2[k] = signbit_convert(real, BITS_PER_SYMBOL);
                    break;
                case 3:
                    im3[k] = signbit_convert(imag, BITS_PER_SYMBOL);
                    re3[k] = signbit_convert(real, BITS_PER_SYMBOL);
                    break;
                }

                bits_left -= BITS_PER_SYMBOL;
                current_data = current_data >> BITS_PER_SYMBOL;
            }
}