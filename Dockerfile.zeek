FROM zeek/zeek:latest

WORKDIR /app

COPY src/zeek /app/src/zeek

RUN mkdir -p /app/data/pcap /app/data/zeek

ENTRYPOINT ["zeek"]
