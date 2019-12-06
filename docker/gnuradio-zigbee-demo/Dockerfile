FROM gnuradio:uhd

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    libuhd-dev

RUN /usr/lib/uhd/utils/uhd_images_downloader.py

COPY build_gr_ieee802_15_4.sh /root/

RUN chmod +x /root/build_gr_ieee802_15_4.sh

RUN /root/build_gr_ieee802_15_4.sh
RUN rm /root/build_gr_ieee802_15_4.sh

COPY ieee802_15_4_OQPSK_PHY.grc /root/grc-blocks/
RUN grcc /root/grc-blocks/ieee802_15_4_OQPSK_PHY.grc
COPY random_socket_udp.py /root/
COPY zigbee.grc /root/
