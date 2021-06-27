from __future__ import print_function
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from datetime import timedelta
from datetime import datetime
import time
import json

with open('test1.json', 'r', encoding="utf8") as json_file:
    load_data = json.load(json_file)


def dateTimeToSeconds(dateTimeObj):
    """return seconds from datetime object"""
    timeToSeconds = timedelta(hours=dateTimeObj.hour, minutes=dateTimeObj.minute)
    return timeToSeconds.total_seconds()


startTime = datetime.strptime(load_data["params"]["startTime"],
                              "%Y-%m-%d %H:%M:%S.%f")


def convertTimeString(time_strings):
    """convert time strings to datatime"""
    convertTime = datetime.strptime(time_strings,
                                    "%Y-%m-%d %H:%M:%S.%f")
    return convertTime


def createDummyDepots(data):
    """create the number of dummy depots"""
    quantity = len(data["vehicles"]) + 3
    return quantity


def append_request(request, requests, n):
    """extacts data from input"""
    earlyTime = datetime.strptime(request["earlyDeliveryTime"],
                                  "%Y-%m-%d %H:%M:%S.%f")
    lateTime = datetime.strptime(request["lateDeliveryTime"],
                                 "%Y-%m-%d %H:%M:%S.%f")
    timeWindows = (max(0, int(dateTimeToSeconds(earlyTime) - dateTimeToSeconds(startTime))),
                   dateTimeToSeconds(lateTime) - dateTimeToSeconds(startTime))
    requests.append([])
    requests[n].append(request["orderCode"])
    requests[n].append(request["deliveryAddr"])
    requests[n].append(request["deliveryLocationCode"])
    requests[n].append((request["deliveryLat"], request["deliveryLng"]))
    requests[n].append(timeWindows)
    requests[n].append((request["earlyDeliveryTime"], request["lateDeliveryTime"]))
    requests[n].append(request["cusNote1"])
    requests[n].append(request["priority"])
    return requests

def maxVehicleCapacity(load_data):
    """return max vehicle capacity"""
    if load_data["params"]["extendCapacity"] > 0:
        max_vehicle_capacity = max([vehicle["weight"] * (1 + (load_data["params"]["extendCapacity"]) / 100)
                                    for vehicle in load_data["vehicles"]])
    else:
        max_vehicle_capacity = max([vehicle["weight"] for vehicle in load_data["vehicles"]])
    return max_vehicle_capacity


def export_data(load_data):
    """stores the data problem as list type"""
    requests = [[0, "Quận Tân Phú", "R1_HCM", (10.761124256132712, 106.63281628678588), (0, 70000), ("", ""),
                 "", "", "", [], [], [], 0, [], []]]
    n = 1
    max_vehicle_capacity = maxVehicleCapacity(load_data)
    for request in load_data["requests"]:
        if sum([item["weight"] for item in request["items"]]) <= max_vehicle_capacity:
            append_request(request, requests, n)
            requests[n].append("")
            requests[n].append([item["weight"] for item in request["items"]])
            requests[n].append([item["cbm"] for item in request["items"]])
            requests[n].append([item["quantity"] for item in request["items"]])
            requests[n].append(sum([item["deliveryDuration"] for item in request["items"]])
                               + request["fixUnloadTime"])
            requests[n].append([item["code"] for item in request["items"]])
            requests[n].append([item["itemType"] for item in request["items"]])
            n += 1
        else:
            new_nodes = []
            cbm = []
            quantity = []
            unloading_time = 0
            code = []
            item_type = []
            for index in range(len(request["items"])):
                if request["items"][index]["weight"] > max_vehicle_capacity:
                    append_request(request, requests, n)
                    requests[n].append("over_weight")
                    requests[n].append([request["items"][index]["weight"]])
                    requests[n].append([request["items"][index]["cbm"]])
                    requests[n].append([request["items"][index]["quantity"]])
                    requests[n].append(request["items"][index]["deliveryDuration"] + request["fixUnloadTime"])
                    requests[n].append([request["items"][index]["code"]])
                    requests[n].append([request["items"][index]["itemType"]])
                    n += 1
                else:
                    pre_nodes = list(new_nodes)
                    pre_cbm = list(cbm)
                    pre_quantity = list(quantity)
                    pre_unloading_time = unloading_time
                    pre_code = list(code)
                    pre_item_type = list(item_type)
                    new_nodes.append(request["items"][index]["weight"])
                    cbm.append(request["items"][index]["cbm"])
                    quantity.append(request["items"][index]["quantity"])
                    unloading_time += request["items"][index]["deliveryDuration"]
                    code.append(request["items"][index]["code"])
                    item_type.append(request["items"][index]["itemType"])
                    if index != (len(request["items"]) - 1):
                        if sum(new_nodes) > max_vehicle_capacity:
                            append_request(request, requests, n)
                            requests[n].append("split_order")
                            requests[n].append(pre_nodes)
                            requests[n].append(pre_cbm)
                            requests[n].append(pre_quantity)
                            requests[n].append(pre_unloading_time + request["fixUnloadTime"])
                            requests[n].append(pre_code)
                            requests[n].append(item_type)
                            n += 1
                            new_nodes = list([request["items"][index]["weight"]])
                            cbm = list([request["items"][index]["cbm"]])
                            quantity = list([request["items"][index]["quantity"]])
                            unloading_time = request["items"][index]["deliveryDuration"]
                            code = list([request["items"][index]["code"]])
                            item_type = list([request["items"][index]["itemType"]])
                    else:
                        if sum(new_nodes) > max_vehicle_capacity:
                            append_request(request, requests, n)
                            requests[n].append("split_order")
                            requests[n].append(pre_nodes)
                            requests[n].append(pre_cbm)
                            requests[n].append(pre_quantity)
                            requests[n].append(pre_unloading_time + request["fixUnloadTime"])
                            requests[n].append(pre_code)
                            requests[n].append(pre_item_type)
                            n += 1
                            append_request(request, requests, n)
                            requests[n].append("split_order")
                            requests[n].append([request["items"][index]["weight"]])
                            requests[n].append([request["items"][index]["cbm"]])
                            requests[n].append([request["items"][index]["quantity"]])
                            requests[n].append(request["items"][index]["deliveryDuration"] + request["fixUnloadTime"])
                            requests[n].append([request["items"][index]["code"]])
                            requests[n].append([request["items"][index]["itemType"]])
                            n += 1
                        else:
                            append_request(request, requests, n)
                            requests[n].append("split_order")
                            requests[n].append(new_nodes)
                            requests[n].append(cbm)
                            requests[n].append(quantity)
                            requests[n].append(unloading_time + request["fixUnloadTime"])
                            requests[n].append(code)
                            requests[n].append(item_type)
                            n += 1
    dummy_depots = list(requests[0])
    dummy_depots[12] = 1800
    num_dummy_depots = createDummyDepots(load_data)
    requests = [requests[0]] + [dummy_depots] * num_dummy_depots + requests[1:]
    return requests

def nodeVehicleRelation(requests):
    """return nodes allowed to vist by one type of vehicle"""
    group_vehicle = {}
    for id, vehicle in enumerate(load_data["vehicles"]):
        if vehicle["group"] not in group_vehicle.keys():
            group_vehicle[vehicle["group"]] = [-1]
            group_vehicle[vehicle["group"]].append(id)
        else:
            group_vehicle[vehicle["group"]].append(id)
    priority_vehicle = []
    for id, request in enumerate(requests):
        if len(request[7]) == 1:
            priority_group = request[7][0].split(',')
            for gr, vehicleId in group_vehicle.items():
                if priority_group[1] == gr:
                    priority_vehicle.append([id, vehicleId])
    return priority_vehicle


def findMutalNodes(requests):
    """return node index, which have the same code"""
    mutal_node_dict = {}
    for index, request in enumerate(requests):
        if request[2] == "R1_HCM" or request[8] == "over_weight":
            continue
        if request[2] not in mutal_node_dict.keys():
            mutal_node_dict[request[2]] = []
            mutal_node_dict[request[2]].append([])
            mutal_node_dict[request[2]].append([])
            mutal_node_dict[request[2]][0].append(index)
            mutal_node_dict[request[2]][1].append(sum(request[9]))
        else:
            mutal_node_dict[request[2]][0].append(index)
            mutal_node_dict[request[2]][1].append(sum(request[9]))
    mutal_nodes = [index for index in mutal_node_dict.values() if len(index[0]) > 1]
    return mutal_nodes


def parseCusNote(cusnote1):
    cusnote = cusnote1.split(';')
    if len(cusnote) == 1:
        return cusnote[0]
    else:
        return cusnote[1]


def location_code(requests):
    locationCode = [[request[2], parseCusNote(request[6])] for request in requests]
    return locationCode


def clusterNodes(locationCode):
    cluster_nodes_dict = {}
    for id, code in enumerate(locationCode):
        if code[0] == "R1_HCM":
            continue
        elif code[1] not in cluster_nodes_dict.keys():
            cluster_nodes_dict[code[1]] = []
            cluster_nodes_dict[code[1]].append(id)
        else:
            cluster_nodes_dict[code[1]].append(id)
    cluster_nodes = [node for node in cluster_nodes_dict.values()]
    return cluster_nodes


def break_time(requests):
    interrupt_periods = {}
    break_intervals = []
    locationCode = location_code(requests)
    for config in load_data["locationConfigs"]:
        interrupt_periods[config["locationCode"]] = config["interupPeriods"]
    for code, cusNote in locationCode:
        if len(interrupt_periods[code]) == 0:
            break_intervals.append((0, 0))
        else:
            start_break = datetime.strptime(interrupt_periods[code][0]["start"],
                                            "%Y-%m-%d %H:%M:%S.%f")
            end_break = datetime.strptime(interrupt_periods[code][0]["end"],
                                          "%Y-%m-%d %H:%M:%S.%f")
            break_intervals.append((dateTimeToSeconds(start_break) - dateTimeToSeconds(startTime),
                                    dateTimeToSeconds(end_break) - dateTimeToSeconds(startTime)))
    return break_intervals


###########################
# Problem Data Definition #
###########################
def create_data_model(requests, load_data):
    """ input """
    data = {}
    # Locations in block unit
    data["num_locations"] = len(requests)
    demand = [sum(requests[i][9]) for i in range(len(requests))]
    data["time_windows"] = [requests[i][4] for i in range(len(requests))]
    data["vehicle_capacity"] = [int(vehicle["weight"] * (1 + load_data["params"]["extendCapacity"] / 100))
                                for vehicle in load_data["vehicles"]]
    data["max_travel_time"] = []
    for vehicle in load_data["vehicles"]:
        if vehicle["endWorkingTime"] == None:
            data["max_travel_time"].append(0)
        else:
            data["max_travel_time"].append(int(dateTimeToSeconds(convertTimeString(vehicle["endWorkingTime"]))
                                               - dateTimeToSeconds(startTime)))
    data["num_vehicles"] = len(data["vehicle_capacity"])
    max_vehicle_capacity1 = max(data["vehicle_capacity"])
    data["num_dummy_depots"] = createDummyDepots(load_data)
    data["demands"] = [demand[0]] + [-max_vehicle_capacity1] * data["num_dummy_depots"] \
                      + demand[(data["num_dummy_depots"] + 1):]
    # data["split_order"] = split_order(requests)
    data["mutal_nodes"] = findMutalNodes(requests)
    data["break_intervals"] = break_time(requests)
    locationCode = location_code(requests)
    data["cluster_nodes"] = clusterNodes(locationCode)
    data["node_vehicle_relations"] = nodeVehicleRelation(requests)
    data["vehicle_speed"] = 30
    data["depot"] = 0
    return data


#######################
# Problem Constraints #
#######################

def create_distance_evaluator(requests):
    """Creates callback to return distance between points."""
    locationCode = []
    for request in requests:
        locationCode.append(request[2])
    # return distance matrix from json input
    distances = {}
    for fromCounter, fromNode in enumerate(locationCode):
        distances[fromCounter] = {}
        for toCounter, toNode in enumerate(locationCode):
            if fromCounter == toCounter:
                distances[fromCounter][toCounter] = 0
            else:
                for item in load_data['distances']:
                    if fromNode == toNode:
                        distances[fromCounter][toCounter] = 0
                    elif item['srcCode'] == fromNode and item['destCode'] == toNode:
                        if (item['srcCode'] == "R1_HCM" or item['destCode'] == "R1_HCM") and (item['distance'] < 2):
                            distances[fromCounter][toCounter] = 2
                        else:
                            distances[fromCounter][toCounter] = item['distance']
    return distances


def create_time_evaluator(requests):
    distances = create_distance_evaluator(requests)
    locationCode = location_code(requests)
    vehicleSpeed = 20
    total_time = {}
    for from_counter, from_node in enumerate(locationCode):
        total_time[from_counter] = {}
        for to_counter, to_node in enumerate(locationCode):
            if from_node[0] == to_node[0]:
                if (from_counter == to_counter) or (from_node[0] != "R1_HCM"):
                    total_time[from_counter][to_counter] = 0
                else:
                    total_time[from_counter][to_counter] = 1000000000
            else:
                total_time[from_counter][to_counter] = int(
                    (distances[from_counter][to_counter] * 3600) / vehicleSpeed +
                    requests[to_counter][12])
    return total_time


def expected_time(seconds, data):
    seconds_to_time = time.strftime("%H:%M:%S", time.gmtime(seconds))
    delivery_datetime = datetime.strptime(data["params"]["startTime"]
                                          , "%Y-%m-%d %H:%M:%S.%f")
    expected_time = "{0} {1}".format(delivery_datetime.date(), seconds_to_time)
    return expected_time


def export_json(load_data, data, manager, routing, assignment, requests):
    """export json output"""
    json_output = {}
    json_output["solutions"] = []
    json_output["solutions"].append({})
    routes = []
    depot = int(createDummyDepots(load_data))
    time_dimension = routing.GetDimensionOrDie('Time')
    m = 0
    load_trips = []
    distance_trips = []
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        pre_index = index
        if routing.IsEnd(assignment.Value(routing.NextVar(index))):
            continue
        else:
            load_trips.append([])
            distance_trips.append([])
            routes.append({})
            routes[m]["vehicle"] = {}
            routes[m]["vehicle"]["group"] = load_data["vehicles"][vehicle_id]["group"]
            routes[m]["vehicle"]["code"] = load_data["vehicles"][vehicle_id]["code"]
            routes[m]["vehicle"]["lat"] = load_data["vehicles"][vehicle_id]["lat"]
            routes[m]["vehicle"]["lng"] = load_data["vehicles"][vehicle_id]["lng"]
            routes[m]["vehicle"]["endLat"] = load_data["vehicles"][vehicle_id]["endLat"]
            routes[m]["vehicle"]["endLng"] = load_data["vehicles"][vehicle_id]["endLng"]
            routes[m]["vehicle"]["startWorkingTime"] = load_data["vehicles"][vehicle_id]["startWorkingTime"]
            routes[m]["vehicle"]["endWorkingTime"] = load_data["vehicles"][vehicle_id]["endWorkingTime"]
            routes[m]["vehicle"]["availableCapacity"] = load_data["vehicles"][vehicle_id]["weight"]
            routes[m]["vehicle"]["availableCbm"] = load_data["vehicles"][vehicle_id]["cbm"]
            n = 0
            load = 0
            route_distance = 0
            routes[m]["elements"] = []
            while not routing.IsEnd(index):
                time_var = time_dimension.CumulVar(index)
                previous_index = manager.IndexToNode(pre_index)
                node_index = manager.IndexToNode(index)
                # next_node_index = manager.IndexToNode(assignment.Value(routing.NextVar(index)))
                if node_index in range(0, depot + 1):
                    distance = 0
                else:
                    distance = routing.GetArcCostForVehicle(previous_index, node_index, vehicle_id)
                route_distance += distance
                routes[m]["elements"].append({})
                routes[m]["elements"][n]["code"] = requests[node_index][2]
                routes[m]["elements"][n]["address"] = requests[node_index][1]
                routes[m]["elements"][n]["lat"] = requests[node_index][3][0]
                routes[m]["elements"][n]["lng"] = requests[node_index][3][1]
                routes[m]["elements"][n]["arrivalTime"] = expected_time(
                    (assignment.Min(time_var) - requests[node_index][12] + dateTimeToSeconds(startTime))
                    , load_data)
                routes[m]["elements"][n]["departureTime"] = expected_time(
                    (assignment.Min(time_var) + dateTimeToSeconds(startTime)), load_data)
                routes[m]["elements"][n]["description"] = ""
                routes[m]["elements"][n]["orderCode"] = requests[node_index][0]
                if data["demands"][node_index] < 0:
                    load_trips[m].append((load / data["vehicle_capacity"][vehicle_id]) * 100)
                    distance_trips[m].append(route_distance - sum(distance_trips[m]))
                    load = 0
                else:
                    load += data["demands"][node_index]
                routes[m]["elements"][n]["load"] = load
                routes[m]["elements"][n]["distance"] = route_distance
                if len(requests[node_index][13]) > 0:
                    routes[m]["elements"][n]["items"] = []
                    for i in range(len(requests[node_index][13])):
                        routes[m]["elements"][n]["items"].append({})
                        routes[m]["elements"][n]["items"][i]["code"] = requests[node_index][13][i]
                        routes[m]["elements"][n]["items"][i]["itemType"] = requests[node_index][14][i]
                        routes[m]["elements"][n]["items"][i]["weight"] = requests[node_index][9][i]
                        routes[m]["elements"][n]["items"][i]["quantity"] = requests[node_index][11][i]
                        routes[m]["elements"][n]["items"][i]["cbm"] = requests[node_index][10][i]
                        routes[m]["elements"][n]["items"][i]["description"] = ""
                pre_index = index
                index = assignment.Value(routing.NextVar(index))
                n += 1
            time_var = time_dimension.CumulVar(index)
            node_index = manager.IndexToNode(index)
            previous_index = manager.IndexToNode(pre_index)
            if node_index in range(0, depot + 1):
                distance = 0
            else:
                distance = routing.GetArcCostForVehicle(previous_index, node_index, vehicle_id)
            route_distance += distance
            routes[m]["elements"].append(dict(routes[m]["elements"][0]))
            routes[m]["elements"][n]["arrivalTime"] = expected_time(
                (assignment.Min(time_var) + dateTimeToSeconds(startTime)), load_data)
            routes[m]["elements"][n]["departureTime"] = expected_time(
                (assignment.Min(time_var) + dateTimeToSeconds(startTime)), load_data)
            routes[m]["elements"][n]["distance"] = route_distance
            load_trips[m].append((load / data["vehicle_capacity"][vehicle_id]) * 100)
            distance_trips[m].append(route_distance - sum(distance_trips[m]))
            routes[m]["distance"] = route_distance
            m += 1
    json_output["solutions"][0]["routes"] = routes
    json_output["solutions"][0]["unScheduledRequests"] = []
    # create a list containing dropped nodes
    dropped_nodes = []
    dummy_depots = createDummyDepots(load_data)
    for node in range(routing.Size()):
        if routing.IsStart(node) or routing.IsEnd(node):
            continue
        if node in range(1, dummy_depots + 1):
            continue
        if assignment.Value(routing.NextVar(node)) == node:
            dropped_nodes.append(manager.IndexToNode(node))
    if len(dropped_nodes) > 0:
        id = 0
        for node in dropped_nodes:
            json_output["solutions"][0]["unScheduledRequests"].append({})
            json_output["solutions"][0]["unScheduledRequests"][id]["orderCode"] = requests[node][0]
            json_output["solutions"][0]["unScheduledRequests"][id]["pickupLocationCode"] = requests[0][2]
            json_output["solutions"][0]["unScheduledRequests"][id]["pickupAddr"] = requests[0][1]
            json_output["solutions"][0]["unScheduledRequests"][id]["pickupLat"] = requests[0][3][0]
            json_output["solutions"][0]["unScheduledRequests"][id]["pickupLng"] = requests[0][3][1]
            json_output["solutions"][0]["unScheduledRequests"][id]["deliveryLocationCode"] = requests[node][2]
            json_output["solutions"][0]["unScheduledRequests"][id]["deliveryAddr"] = requests[node][1]
            json_output["solutions"][0]["unScheduledRequests"][id]["deliveryLat"] = requests[node][3][0]
            json_output["solutions"][0]["unScheduledRequests"][id]["deliveryLng"] = requests[node][3][1]
            json_output["solutions"][0]["unScheduledRequests"][id]["earlyDeliveyTime"] = requests[node][5][0]
            json_output["solutions"][0]["unScheduledRequests"][id]["lateDeliveyTime"] = requests[node][5][1]
            if len(requests[node][13]) > 0:
                json_output["solutions"][0]["unScheduledRequests"][id]["items"] = []
                for i in range(len(requests[node][13])):
                    json_output["solutions"][0]["unScheduledRequests"][id]["items"].append({})
                    json_output["solutions"][0]["unScheduledRequests"][id]["items"][i]["code"] = requests[node][13][i]
                    json_output["solutions"][0]["unScheduledRequests"][id]["items"][i]["itemType"] = requests[node][14][
                        i]
                    json_output["solutions"][0]["unScheduledRequests"][id]["items"][i]["weight"] = requests[node][9][i]
                    json_output["solutions"][0]["unScheduledRequests"][id]["items"][i]["quatity"] = requests[node][11][
                        i]
                    json_output["solutions"][0]["unScheduledRequests"][id]["items"][i]["cbm"] = requests[node][10][i]
                    json_output["solutions"][0]["unScheduledRequests"][id]["items"][i]["description"] = ""
            id += 1

    json_output["solutions"][0]["statistics"] = {}
    json_output["solutions"][0]["statistics"]["totalDistance"] = sum([sum(trip_distance)
                                                                      for trip_distance in distance_trips])
    json_output["solutions"][0]["statistics"]["numVehicles"] = len(distance_trips)
    json_output["solutions"][0]["statistics"]["numTrips"] = sum([len(trip)
                                                                 for trip in distance_trips])
    json_output["solutions"][0]["statistics"]["longestTrip"] = max([max(trip)
                                                                    for trip in distance_trips])
    json_output["solutions"][0]["statistics"]["shortestTrip"] = min([min(trip)
                                                                     for trip in distance_trips])
    json_output["solutions"][0]["statistics"]["averageTripDistance"] = \
        json_output["solutions"][0]["statistics"]["totalDistance"] / json_output["solutions"][0]["statistics"][
            "numTrips"]

    json_output["solutions"][0]["statistics"]["maximumTransferRatio"] = max([max(trip)
                                                                             for trip in load_trips])
    json_output["solutions"][0]["statistics"]["minimumTransferRatio"] = min([min(trip)
                                                                             for trip in load_trips])
    json_output["solutions"][0]["statistics"]["averageTransferRatio"] = \
        sum([sum(trip) for trip in load_trips]) / sum([len(trip) for trip in distance_trips])
    json_output["solutions"][0]["statistics"]["distanceTripList"] = distance_trips
    json_output["solutions"][0]["statistics"]["loadTripList"] = load_trips
    return json_output


########
# Main #
########
def main():
    """Entry point of the program"""
    # Instantiate the data problem.
    requests = export_data(load_data)
    data = create_data_model(requests, load_data)
    manager = pywrapcp.RoutingIndexManager(data["num_locations"], data['num_vehicles'], data['depot'])
    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)
    # Define weight of each edge
    distance_matrix = create_distance_evaluator(requests)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,
        100,
        True,
        dimension_name)

    distance_dimension = routing.GetDimensionOrDie(dimension_name)

    # Add Capacity constraint
    def demand_evaluator(from_index):
        """Returns the demand of the current node"""
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_evaluator)
    vehicle_capacity = data["vehicle_capacity"]
    for capacity in vehicle_capacity:
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            capacity,  # Null slack
            vehicle_capacity,
            True,  # start cumul to zero
            'capacity')
    capacity_dimension = routing.GetDimensionOrDie('capacity')
    dummy_depots = createDummyDepots(load_data)
    for node_index in range(routing.nodes()):
        index = manager.NodeToIndex(node_index)
        if node_index not in range(1, dummy_depots + 1):
            capacity_dimension.SlackVar(index).SetValue(0)
        routing.AddVariableMinimizedByFinalizer(capacity_dimension.CumulVar(index))

    time_matrix = create_time_evaluator(requests)

    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return time_matrix[from_node][to_node]

    travel_time_callback_index = routing.RegisterTransitCallback(time_callback)
    penalty = 500
    for node in range(1, len(distance_matrix)):
        if node in range(1, data["num_dummy_depots"] + 1):
            routing.AddDisjunction([manager.NodeToIndex(node)], 3)
        else:
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)
    """Add Time windows constraint"""
    time = 'Time'
    horizon = 40000
    routing.AddDimension(
        travel_time_callback_index,
        1800,
        horizon,
        False,
        time)
    time_dimension = routing.GetDimensionOrDie(time)
    for location_idx, time_window in enumerate(data["time_windows"]):
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(int(time_window[0]), int(time_window[1]))

    for location_idx, break_interval in enumerate(data["break_intervals"]):
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).RemoveInterval(int(break_interval[0]), int(break_interval[1]))
        routing.AddToAssignment(time_dimension.SlackVar(index))

    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(data["time_windows"][0][0],
                                                data["time_windows"][0][1])
        routing.AddToAssignment(time_dimension.SlackVar(index))

    for i in range(data['num_vehicles']):
        routing.AddVariableMinimizedByFinalizer(
            distance_dimension.CumulVar(routing.Start(i)))
        routing.AddVariableMinimizedByFinalizer(
            distance_dimension.CumulVar(routing.End(i)))

    for relation in data["node_vehicle_relations"]:
        index = manager.NodeToIndex(relation[0])
        routing.VehicleVar(index).SetValues(relation[1])

    for nodes in data['mutal_nodes']:
        routing.AddSoftSameVehicleConstraint(nodes[0], 50)
    #
    for cluster in data["cluster_nodes"]:
        routing.AddSoftSameVehicleConstraint(cluster, 50)
    # add allowed travel time contraint for vehicles
    solver = routing.solver()
    for vehicle, value in enumerate(data["max_travel_time"]):
        id = manager.NodeToIndex(vehicle)
        if value == 0:
            solver.AddConstraint(time_dimension.CumulVar(routing.End(id)) <= horizon)
        else:
            solver.AddConstraint(time_dimension.CumulVar(routing.End(id)) <= value)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH)
    search_parameters.time_limit.seconds = 300
    search_parameters.log_search = True
    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)
    if assignment:
        # print_solution(data, manager, routing, assignment, requests)
        output = export_json(load_data, data, manager, routing, assignment, requests)
        with open('output.json', 'w', encoding="utf8") as outfile:
            json.dump(output, outfile)
    else:
        print('No solution has been found')


if __name__ == '__main__':
    main()
