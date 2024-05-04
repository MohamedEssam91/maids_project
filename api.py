import pandas as pd
from flask import Flask, request, jsonify
from model import load_data, preprocess_data, train_model

app = Flask(__name__)

# Load or initialize devices DataFrame
try:
    devices_df = load_data('devices.xlsx')
except FileNotFoundError:
    devices_df = pd.DataFrame(columns=[
        "id", "battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g", 
        "int_memory", "m_dep", "mobile_wt", "n_cores", "pc", "px_height", 
        "px_width", "ram", "sc_h", "sc_w", "talk_time", "three_g", 
        "touch_screen", "wifi", "price_range"
    ])

@app.route('/api/devices', methods=['GET'])
def get_devices():
    """Retrieve details of all devices."""
    return jsonify(devices_df.to_dict('records'))

@app.route('/api/devices/<int:device_id>', methods=['GET'])
def get_device(device_id):
    """Retrieve details of a specific device by ID."""
    device = devices_df.loc[devices_df['id'] == device_id].to_dict('records')
    if device:
        return jsonify(device[0])
    else:
        return jsonify({"error": "Device not found"}), 404

@app.route('/api/devices', methods=['POST'])
def add_device():
    """Add a new device to the database."""
    global devices_df
    data = request.get_json()
    new_device_id = devices_df['id'].max() + 1 if not devices_df.empty else 1
    new_device = {"id": new_device_id, **data}
    new_device_df = pd.DataFrame([new_device])
    devices_df = pd.concat([devices_df, new_device_df], ignore_index=True)
    return jsonify({"message": "Device added successfully"}), 201

@app.route('/api/predict/<int:device_id>', methods=['POST'])
def predict_price(device_id):
    """Predict price for a specific device."""
    global devices_df
    device = devices_df.loc[devices_df['id'] ==  device_id]
    if not device.empty:
        # Preprocess device data
        device_data, _ = preprocess_data(device)
        # Make prediction
        predicted_price_range = model.predict(device_data)
        devices_df.loc[devices_df['id'] == device_id, 'price_range'] = predicted_price_range[0]
        return jsonify({"message": f"Price predicted and updated successfully and the new price is {predicted_price_range[0]}"}), 200
    else:
        return jsonify({"error": "Device not found"}), 404

if __name__ == '__main__':
    df = load_data('train_maids.xlsx')
    x_train, y_train = preprocess_data(df)
    model = train_model(x_train, y_train)
    app.run(debug=True)
