/* $ gcc bladerf_test.c -o bladerf_test -lbladeRF
 * C:\Program Files\bladeRF\include
 * C:\Program Files\bladeRF\x64
 * 
 * cl /Zi /EHsc -I "C:\Program Files\bladeRF\include" bladerf_test.c /link bladeRF.lib /LIBPATH:"C:\Program Files\bladeRF\x64" /Fe"bladerf_test"
 * specify /MD for shared library
 */
 
#include <libbladeRF.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* The RX and TX channels are configured independently for these parameters */
struct channel_config {
    bladerf_channel channel;
    unsigned int frequency;
    unsigned int bandwidth;
    unsigned int samplerate;
    int gain;
};

int configure_channel(struct bladerf *dev, struct channel_config *c)
{
    int status;
 
    status = bladerf_set_frequency(dev, c->channel, c->frequency);
    if (status != 0) {
        fprintf(stderr, "Failed to set frequency = %u: %s\n", c->frequency,
                bladerf_strerror(status));
        return status;
    }
 
    status = bladerf_set_sample_rate(dev, c->channel, c->samplerate, NULL);
    if (status != 0) {
        fprintf(stderr, "Failed to set samplerate = %u: %s\n", c->samplerate,
                bladerf_strerror(status));
        return status;
    }
 
    status = bladerf_set_bandwidth(dev, c->channel, c->bandwidth, NULL);
    if (status != 0) {
        fprintf(stderr, "Failed to set bandwidth = %u: %s\n", c->bandwidth,
                bladerf_strerror(status));
        return status;
    }
 
    status = bladerf_set_gain(dev, c->channel, c->gain);
    if (status != 0) {
        fprintf(stderr, "Failed to set gain: %s\n", bladerf_strerror(status));
        return status;
    }
 
    return status;
}
 
/* Usage:
 *   libbladeRF_example_boilerplate [serial #]
 *
 * If a serial number is supplied, the program will attempt to open the
 * device with the provided serial number.
 *
 * Otherwise, the first available device will be used.
 */
int main(int argc, char *argv[])
{
    int status;
    struct bladerf_metadata meta;
    struct channel_config config;
    struct bladerf_range *rf_range;
 
    struct bladerf *dev = NULL;
    struct bladerf_devinfo dev_info;
    
    printf("[INFO] Initialise bladeRF\n");

    /* Initialize the information used to identify the desired device
     * to all wildcard (i.e., "any device") values */
    bladerf_init_devinfo(&dev_info);

    /* Request a device with the provided serial number.
     * Invalid strings should simply fail to match a device. */
    if (argc >= 2) {
        strncpy(dev_info.serial, argv[1], sizeof(dev_info.serial) - 1);
    }
 
    status = bladerf_open_with_devinfo(&dev, &dev_info);
    if (status != 0) {
        fprintf(stderr, "Unable to open device: %s\n",
                bladerf_strerror(status));
 
        return 1;
    }

    printf("[INFO] bladeRF enable oversampling\n");
    bladerf_enable_feature(dev, (bladerf_feature)1, true);
    
    printf("[INFO] bladeRF configure RX(0)\n");
 
    /* Set up RX channel parameters */
    config.channel    = BLADERF_CHANNEL_RX(0);
    config.frequency  = 2'410'000'000;
    config.bandwidth  =     5'000'000;
    config.samplerate =    10'000'000;
    config.gain       =            39;
 
    status = configure_channel(dev, &config);
    if (status != 0) {
        fprintf(stderr, "Failed to configure RX channel. Exiting.\n");
        goto out;
    }

    printf("[INFO] enable bladeRF RX(0)\n");
    bladerf_enable_module(dev, config.channel, true);

    printf("[INFO] get RX(0) frequency range\n");
    bladerf_get_frequency_range(dev, config.channel, &rf_range);
    printf("min: %lld, max: %lld, step: %lld, scale:%f\n", rf_range->min, rf_range->max, rf_range->step, rf_range->scale);

    printf("[INFO] get RX(0) sample rate range\n");
    bladerf_get_sample_rate_range(dev, config.channel, &rf_range);
    printf("min: %lld, max: %lld, step: %lld, scale:%f\n", rf_range->min, rf_range->max, rf_range->step, rf_range->scale);
 
    printf("[INFO] RX stream buffer\n");

    unsigned int timeout_ms = 1'000;
    unsigned int buffer_size = 1'024;
    int8_t samples[1'024];
    /*
    bladerf_init_stream(struct bladerf_stream **stream,
        dev,
        bladerf_stream_cb callback,
        void ***buffers,
        size_t num_buffers,
        bladerf_format format,
        size_t samples_per_buffer,
        size_t num_transfers,
        void *user_data);
    */
    /*
    bladerf_sync_rx(dev,
            (void *)&samples,
            buffer_size,
            &meta,
            timeout_ms);
    */
    /*
    bladerf_deinterleave_stream_buffer((bladerf_channel_layout)2,
        (bladerf_format)3,
        buffer_size,
        (void *)&samples);
    */
    //printf("RX[0]: %d + i%d\n", samples[0], samples[1]);
 
    /*
    printf("[INFO] bladeRF configure TX(0)\n");

    // Set up TX channel parameters
    config.channel    = BLADERF_CHANNEL_TX(0);
    config.frequency  = 918'000'000;
    config.bandwidth  =   5'000'000;
    config.samplerate =  10'000'000;
    config.gain       = -14;
 
    status = configure_channel(dev, &config);
    if (status != 0) {
        fprintf(stderr, "Failed to configure TX channel. Exiting.\n");
        goto out;
    }
    */
 
out:
    printf("[INFO] close bladeRF\n");

    bladerf_close(dev);
    return status;
}