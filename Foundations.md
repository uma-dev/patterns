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


# Multi-class Classification Dataset

### a) Dataset name  
**Tech Students - profile prediction**


### b) Number of patterns (entries)
157,800

### c) Patterns per class  
- **Normal:** 6,024  
- **Attack:** 32,025  

### d) Attributes and types  
- 15 attributes (excluding ID/Name)  
- **Numeric (hours, number of courses, average scores)**  
- **Categorical (profile label)**  

### e) Class attribute position  
**Last column** (`PROFILE`)

### f) Example pattern (3rd attribute explained)  
Example:

| NAME   | USER_ID | HOURS_DATASCIENCE | HOURS_BACKEND | HOURS_FRONTEND | NUM_COURSES_BEGINNER_DATASCIENCE | NUM_COURSES_BEGINNER_BACKEND | NUM_COURSES_BEGINNER_FRONTEND | NUM_COURSES_ADVANCED_DATASCIENCE | NUM_COURSES_ADVANCED_BACKEND | NUM_COURSES_ADVANCED_FRONTEND | AVG_SCORE_DATASCIENCE | AVG_SCORE_BACKEND | AVG_SCORE_FRONTEND | PROFILE       |
|--------|---------|-------------------|---------------|----------------|----------------------------------|------------------------------|-------------------------------|---------------------------------|-----------------------------|-------------------------------|----------------------|------------------|-------------------|---------------|
| John   | 123     | 40                | 10            | 5              | 3                                | 1                            | 0                             | 1                               | 0                           | 0                             | 85                   | 70               | 65                | Data Science  |
| Maria  | 124     | 5                 | 35            | 10             | 0                                | 4                            | 1                             | 0                               | 2                           | 0                             | 60                   | 88               | 72                | Backend       |
| Alex   | 125     | 8                 | 12            | 45             | 1                                | 1                            | 3                             | 0                               | 0                           | 2                             | 67                   | 74               | 90                | Frontend      |


- 3rd attribute = **HOURS_DATASCIENCE** → number of hours studied in Data Science.  

### g) Class label of example pattern  
Example profile: **Data Science**  
