import json
from datetime import datetime
import pytz

# The name of the file containing the JSON data
FILE_NAME = "meta_output.txt"

# The timezone for Panama
panama_tz = pytz.timezone('America/Panama')


def process_json_data(file_path):
    """
    Reads a file containing one JSON object per line, processes the data
    to extract object type, colors (including upper/lower for people),
    direction, speed, and time information.

    Args:
        file_path (str): The path to the input file.

    Returns:
        list: A list of dictionaries, where each dictionary is a
              processed event.
    """
    processed_data = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Skip empty lines
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)

                    # Extract object type and colors
                    vehicle_type = "Unknown"
                    color_info = {}

                    if "classes" in data and data["classes"]:
                        # Sort classes by score in descending order
                        sorted_classes = sorted(data["classes"], key=lambda x: x["score"], reverse=True)

                        # Get the top class
                        top_class = sorted_classes[0]
                        vehicle_type = top_class["type"]

                        # Check if the top class is "Vehicle" and if there's a second class
                        if vehicle_type == "Vehicle" and len(sorted_classes) > 1:
                            # Use the next most probable class
                            second_class = sorted_classes[1]
                            vehicle_type = second_class["type"]

                        # Skip the entry if the type is "Unknown" or "Vehicle"
                        if vehicle_type == "Unknown" or vehicle_type == "Vehicle":
                            continue

                        if vehicle_type == "Human":
                            # For Human objects, extract upper and lower clothing colors
                            upper_color = "Unknown"
                            if "upper_clothing_colors" in top_class and top_class["upper_clothing_colors"]:
                                top_upper = max(top_class["upper_clothing_colors"], key=lambda x: x["score"])
                                upper_color = top_upper["name"]

                            lower_color = "Unknown"
                            if "lower_clothing_colors" in top_class and top_class["lower_clothing_colors"]:
                                top_lower = max(top_class["lower_clothing_colors"], key=lambda x: x["score"])
                                lower_color = top_lower["name"]

                            color_info = {
                                "upper_clothing_color": upper_color,
                                "lower_clothing_color": lower_color
                            }
                        else:
                            # For other objects (Car, Truck, etc.), extract the main color
                            main_color = "Unknown"
                            if "colors" in top_class and top_class["colors"]:
                                top_color = max(top_class["colors"], key=lambda x: x["score"])
                                main_color = top_color["name"]

                            color_info = {"main_color": main_color}

                    # Convert timestamps to Panama time (UTC-5)
                    start_time_utc = datetime.fromisoformat(data["start_time"].replace("Z", "+00:00"))
                    end_time_utc = datetime.fromisoformat(data["end_time"].replace("Z", "+00:00"))

                    start_time_panama = start_time_utc.astimezone(panama_tz).strftime("%Y-%m-%d %H:%M:%S")
                    end_time_panama = end_time_utc.astimezone(panama_tz).strftime("%Y-%m-%d %H:%M:%S")

                    # Calculate direction and speed
                    direction = "N/A"
                    speed_percent_sec = "N/A"
                    if "observations" in data and len(data["observations"]) > 1:
                        first_obs = data["observations"][0]["bounding_box"]
                        last_obs = data["observations"][-1]["bounding_box"]

                        # Calculate the center x-coordinate of the bounding boxes
                        start_x_center = (first_obs["left"] + first_obs["right"]) / 2
                        end_x_center = (last_obs["left"] + last_obs["right"]) / 2

                        # Determine direction
                        if end_x_center > start_x_center:
                            direction = "Left-to-Right"
                        else:
                            direction = "Right-to-Left"

                        # Calculate speed in % per second
                        duration = data.get("duration", 0)
                        if duration > 0:
                            horizontal_change = abs(end_x_center - start_x_center)
                            speed = horizontal_change / duration * 100
                            speed_percent_sec = f"{speed:.2f}%/sec"

                    # Create the dictionary for the processed event
                    result = {
                        "type": vehicle_type,
                        "id": data.get("id"),
                        **color_info,  # Unpack the color information
                        "direction": direction,
                        "speed": speed_percent_sec,
                        "start_time": start_time_panama,
                        "end_time": end_time_panama,
                        "duration": data.get("duration", 0)
                    }
                    processed_data.append(result)

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line.strip()}. Error: {e}")
                except KeyError as e:
                    print(f"Missing key in JSON object: {e}")
    except FileNotFoundError:
        print(f"Error: The file '{FILE_NAME}' was not found.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

    return processed_data


def post_process_data(data_list):
    """
    Renames the direction field for readability.
    """
    for item in data_list:
        if "direction" in item:
            if item["direction"] == "Right-to-Left":
                item["direction"] = "Via España a La Pulida"
            elif item["direction"] == "Left-to-Right":
                item["direction"] = "La Pulida a Via España"
    return data_list


def summarize_data(data_list):
    """
    Summarizes the processed data, counting objects by type and direction,
    and including the start, end, and duration of the observation window.
    It also ensures that each object is counted only once by its ID.
    """
    summary = {}
    first_start_time = None
    last_end_time = None
    seen_ids = set()

    # First, find the overall start and end times
    for item in data_list:
        obj_id = item.get("id")
        if obj_id not in seen_ids:
            seen_ids.add(obj_id)

            current_start = datetime.strptime(item.get("start_time"), "%Y-%m-%d %H:%M:%S")
            current_end = datetime.strptime(item.get("end_time"), "%Y-%m-%d %H:%M:%S")

            if first_start_time is None or current_start < first_start_time:
                first_start_time = current_start
            if last_end_time is None or current_end > last_end_time:
                last_end_time = current_end

    # Reset seen_ids for the summary count
    seen_ids = set()

    # Then, count the unique objects
    for item in data_list:
        obj_type = item.get("type")
        direction = item.get("direction")
        obj_id = item.get("id")

        if obj_id in seen_ids:
            continue
        seen_ids.add(obj_id)

        # Filter out unknown types for the summary
        if obj_type == "Unknown":
            continue

        if obj_type and direction:
            if obj_type not in summary:
                summary[obj_type] = {
                    "Via España a La Pulida": 0,
                    "La Pulida a Via España": 0,
                }

            # Increment the counter for the specific type and direction
            if direction == "Via España a La Pulida":
                summary[obj_type]["Via España a La Pulida"] += 1
            elif direction == "La Pulida a Via España":
                summary[obj_type]["La Pulida a Via España"] += 1

    # Calculate total duration based on first and last timestamps
    total_duration_seconds = (
                last_end_time - first_start_time).total_seconds() if first_start_time and last_end_time else 0

    # Convert total duration to hours, minutes, and seconds
    hours = int(total_duration_seconds // 3600)
    minutes = int((total_duration_seconds % 3600) // 60)
    seconds = round(total_duration_seconds % 60, 2)
    formatted_duration = f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"

    # Add observation window metadata to the summary
    summary["Observation Window"] = {
        "start_time": first_start_time.strftime("%Y-%m-%d %H:%M:%S") if first_start_time else "N/A",
        "end_time": last_end_time.strftime("%Y-%m-%d %H:%M:%S") if last_end_time else "N/A",
        "total_duration": formatted_duration
    }

    return summary


if __name__ == "__main__":
    # Process the raw data from the file
    initial_output = process_json_data(FILE_NAME)

    # Apply the post-processing transformations
    processed_output = post_process_data(initial_output)

    # Generate the summary of the data
    summary_output = summarize_data(processed_output)

    # Print the final list of dictionaries as a formatted JSON array
    print("--- Processed Events ---")
    print(json.dumps(processed_output, indent=2, ensure_ascii=False))

    # Print the summary as a separate formatted JSON object
    print("\n\n--- Summary ---")
    print(json.dumps(summary_output, indent=2, ensure_ascii=False))
