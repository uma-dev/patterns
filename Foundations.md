# 2 – Binary Classification Dataset

### a) Dataset name  
**Edge-IIoTset**  

### b) Number of patterns (entries)
157,800

### c) Patterns per class  
- **Normal:** 6,024  
- **Attack:** 32,025  

### d) Attributes and types  
61 attributes – mix of **numeric (continuous)** and **categorical**  

### e) Class attribute position  
**Last column** (`attack_state`) which is "0" for normal traffic and "1" for attack traffic

### f) Example pattern (3rd attribute explained)  
Example:
| frame.time | ip.src_host   | ip.dst_host | arp.dst.proto_ipv4 | arp.opcode | arp.hw.size | arp.src.proto_ipv4 | icmp.checksum | icmp.seq_le | icmp.transmit_timestamp | ... | mqtt.proto_len | mqtt.protoname | mqtt.topic | mqtt.topic_len | mqtt.ver | mbtcp.len | mbtcp.trans_id | mbtcp.unit_id | Attack_label | Attack_type |
|------------|---------------|-------------|---------------------|------------|-------------|---------------------|---------------|-------------|-------------------------|-----|----------------|----------------|------------|----------------|----------|-----------|----------------|---------------|--------------|-------------|
| 6.0        | 192.168.0.152 | 0.0         | 0.0                 | 0.0        | 0.0         | 0.0                 | 0.0           | 0.0         | 0.0                     | ... | 0.0            | 0.0            | 0.0        | 0.0            | 0.0      | 0.0       | 0.0            | 0.0           | 1            | MITM        |

- 3rd attribute = **ip.dst_hos** → e.g. 192.168.0.152 means the destination host IP address is

### g) Class label of example pattern  
**0 → Attack traffic**
