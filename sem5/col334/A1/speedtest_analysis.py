import dpkt
import argparse
import socket
import matplotlib.pyplot as plt
import datetime
from collections import Counter,defaultdict


SERVER_IP_U = ''
CLIENT_IP = ''
t0 = float('inf')

def find_client_server_ips(pcap_file):
    global SERVER_IP_U, CLIENT_IP,t0
    with open(pcap_file, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        ip_counter = Counter()

        for timestamp, buf in pcap:
            eth = dpkt.ethernet.Ethernet(buf)
            if not isinstance(eth.data, dpkt.ip.IP):
                continue

            ip = eth.data
            src_ip = socket.inet_ntoa(ip.src)
            dst_ip = socket.inet_ntoa(ip.dst)
            
            ip_counter[src_ip] += 1
            ip_counter[dst_ip] += 1
            t0 = min(t0,timestamp)

        # Find the two most common IPs
        common_ips = ip_counter.most_common(2)
        if len(common_ips) < 2:
            raise ValueError("Not enough IP addresses found in the pcap file.")

        CLIENT_IP = common_ips[0][0]
        SERVER_IP_U = common_ips[1][0]
        # print(t0==1722520913.035402+44.241553521)
        # print(type(client_ip))

def print_pcap_columns(file_name):
    c1=0
    c2=0
    c3=0
    c4=0
    with open(file_name, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        print(f"{'Timestamp':<20} {'Source IP':<20} {'Destination IP':<20} {'Protocol':<10} {'Length':<10} {'Source Port':<15} {'Destination Port':<15}")
        print("="*110)
        for timestamp, buf in pcap:
            eth = dpkt.ethernet.Ethernet(buf)
            if not isinstance(eth.data, dpkt.ip.IP):
                continue  # Not an IP packet

            ip = eth.data
            src_ip = socket.inet_ntoa(ip.src)
            dst_ip = socket.inet_ntoa(ip.dst)
            length = ip.len
            proto = ip.p

            if isinstance(ip.data, dpkt.tcp.TCP):
                protocol = 'TCP'
                tcp = ip.data
                src_port = tcp.sport
                dst_port = tcp.dport
            elif isinstance(ip.data, dpkt.udp.UDP):
                protocol = 'UDP'
                udp = ip.data
                src_port = udp.sport
                dst_port = udp.dport
            else:
                protocol = 'Other'
                src_port = '-'
                dst_port = '-'
                
            # l = [SERVER_IP_U,CLIENT_IP]
            
            # if src_ip in l or dst_ip in l:    
            #     if src_ip == SERVER_IP_U or dst_ip == SERVER_IP_U:
            #         c2 = c2+1
            #     if src_ip == CLIENT_IP or dst_ip == CLIENT_IP:
            #         c3 = c3+1
            # else:
            #     c4 = c4+1
            
            print(f"{timestamp-t0:<20} {src_ip:<20} {dst_ip:<20} {protocol:<10} {length:<10} {src_port:<15} {dst_port:<15}")
    # print(c1,c2,c3,c4)
                
def parse_pcap(file_name):
    with open(file_name, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        download_packets = []
        upload_packets = []
        background_traffic_bytes = 0

        for timestamp, buf in pcap:
            timestamp = timestamp
            eth = dpkt.ethernet.Ethernet(buf)
            if not isinstance(eth.data, dpkt.ip.IP):
                continue
            ip = eth.data
            src_ip = socket.inet_ntoa(ip.src)
            dst_ip = socket.inet_ntoa(ip.dst)

            if isinstance(ip.data, dpkt.tcp.TCP) or isinstance(ip.data, dpkt.udp.UDP):
                if src_ip == SERVER_IP_U and dst_ip == CLIENT_IP :  # Download direction
                    download_packets.append((timestamp-t0, len(buf)))
                elif src_ip == CLIENT_IP and dst_ip == SERVER_IP_U :  # Upload direction
                    upload_packets.append((timestamp-t0, len(buf)))
                else:
                    background_traffic_bytes += len(buf)

        return download_packets, upload_packets, background_traffic_bytes

def plot_throughput(packets, title):
    times = [ts for ts, _ in packets]
    sizes = [size for _, size in packets]

    plt.figure(figsize=(10, 5))
    plt.plot(times, sizes, label=title)
    plt.xlabel('Time (s)')
    plt.ylabel('Throughput (bytes)')
    plt.title(f'Time-Series of Observed Throughput: {title}')
    plt.legend()
    plt.show()
    
def filter_packets_by_interval(packets):
    start_time = min(pkt[0] for pkt in packets)
    data = defaultdict(int)
    for t,d in packets:
        interval = int(t-start_time)
        data[interval]+= d
    filtered = []
    for k in data:
        value = data[k]
        
        if((value*8/(1e6))>=1):
            filtered.append((k,value))
    return filtered
        

def calculate_average_speed(packets):
    if not packets:
        return 0
    total_bytes = sum(size for _, size in packets)
    total_time = len(packets)  
    # print(total_time)
    total_time = total_time if total_time > 0 else 1  
    speed_mbps = (total_bytes * 8) / (total_time * 1_000_000)  
    return speed_mbps

def calculate_background_ratio(background_bytes, download_packets, upload_packets):
    speed_test_bytes = sum(size for _, size in download_packets + upload_packets)
    if speed_test_bytes == 0:
        return float('inf')  
    # print(background_bytes,speed_test_bytes)
    return speed_test_bytes/(speed_test_bytes+background_bytes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze NDT7 speed test PCAP.')
    parser.add_argument('pcap_file', type=str, help='Path to the input PCAP file.')
    parser.add_argument('--plot', action='store_true', help='Plot time-series of observed throughput.')
    parser.add_argument('--throughput', action='store_true', help='Output the average download and upload speeds.')
    parser.add_argument('--background', action='store_true', help='Calculate the ratio of background traffic to speed test traffic.')
    args = parser.parse_args()

    find_client_server_ips(args.pcap_file)
    download_packets, upload_packets, background_traffic_bytes = parse_pcap(args.pcap_file)
    
    download_packets1 = filter_packets_by_interval(download_packets)
    upload_packets1 = filter_packets_by_interval(upload_packets)

    if args.plot:
        plot_throughput(upload_packets, 'Upload')
        plot_throughput(download_packets, 'Download')

    if args.throughput:
        avg_upload_speed = calculate_average_speed(upload_packets1)
        avg_download_speed = calculate_average_speed(download_packets1)
        print(f'{avg_upload_speed:.2f} Mbps, {avg_download_speed:.2f} Mbps')

    if args.background:
        background_ratio = calculate_background_ratio(background_traffic_bytes, download_packets1, upload_packets1)
        print(f'Speed Test traffic ratio: {background_ratio:.2f}')