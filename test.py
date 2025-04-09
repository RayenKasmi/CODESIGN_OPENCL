def compute_row_distribution(N, throughput_dict):
    """Compute row distribution based on throughput."""
    total_throughput = sum(throughput_dict.values())
    distribution = {}
    #starting with the lowest throught put device
    for device, throughput in sorted(throughput_dict.items(), key=lambda x: x[1]):
        # Calculate the number of rows for this device based on its throughput
        num_rows = int(N * (throughput / total_throughput))
        num_rows = round(num_rows)
        distribution[device] = num_rows
    id_device_lowest_throughput = min(distribution, key=distribution.get)
    id_device_highest_throughput = max(distribution, key=distribution.get)
    # Adjust the distribution to ensure it sums to N
    if(sum(distribution.values()) < N):
        distribution[id_device_highest_throughput] += N - sum(distribution.values())
    elif(sum(distribution.values()) > N):
        distribution[id_device_lowest_throughput] -= sum(distribution.values()) - N 
    leftover = 0

    #start with the lowest distribution device to the highest distribution device
    for device in sorted(distribution, key=distribution.get):
        if(device == id_device_highest_throughput):
            distribution[device] += leftover
        else:
            leftover += distribution[device]%16
            distribution[device] -= distribution[device]%16

    return distribution

throughputs = {
    '0': 137,
    '1': 2.5,
    '2': 19,
}

print(compute_row_distribution(8192, throughputs))