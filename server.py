import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, render_template, request, jsonify
from utils.detection import (
    detect_objects, build_narration, get_ondemand_distance,
    update_last_scan, answer_spatial_query
)
import cv2
import numpy as np
import base64
import requests as req

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        img_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'status': 'error', 'error': 'Could not decode image'})
        lang = data.get('lang', 'en')
        detections, _ = detect_objects(frame)
        narration = build_narration(detections, lang=lang)

        update_last_scan(detections, frame)

        return jsonify({'status': 'ok', 'narration': narration, 'detections': detections})
    except Exception as e:
        print(f"DETECT ERROR: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return jsonify({'status': 'error', 'error': str(e)})

@app.route('/geocode', methods=['POST'])
def geocode():
    try:
        data = request.json
        place = data.get('place', '')
        headers = {'User-Agent': 'AIEyes-UAE/1.0 (accessibility project)'}
        res = req.get(
            'https://nominatim.openstreetmap.org/search',
            params={'q': place, 'format': 'json', 'limit': 3},
            headers=headers, timeout=10
        )
        results = res.json()
        if not results:
            return jsonify({'status': 'error', 'message': f'Could not find: {place}'})
        best = results[0]
        return jsonify({
            'status': 'ok',
            'lat': float(best['lat']),
            'lng': float(best['lon']),
            'name': best.get('display_name', place).split(',')[0]
        })
    except Exception as e:
        print(f"GEOCODE ERROR: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/distance-check', methods=['POST'])
def distance_check():
    try:
        data = request.json
        img_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'status': 'error', 'error': 'Could not decode image'})
        lang = data.get('lang', 'en')

        detections, _ = detect_objects(frame)
        narration = get_ondemand_distance(frame, detections, lang=lang)

        return jsonify({'status': 'ok', 'narration': narration})
    except Exception as e:
        print(f"DISTANCE CHECK ERROR: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return jsonify({'status': 'error', 'error': str(e)})

@app.route('/spatial-query', methods=['POST'])
def spatial_query():
    try:
        data = request.json
        query_type = data.get('query_type')
        lang = data.get('lang', 'en')
        answer = answer_spatial_query(query_type, lang=lang)
        return jsonify({'status': 'ok', 'narration': answer})
    except Exception as e:
        print(f"SPATIAL QUERY ERROR: {e}")
        return jsonify({'status': 'error', 'error': str(e)})

@app.route('/navigate', methods=['POST'])
def navigate():
    try:
        data = request.json
        user_lat = float(data['user_lat'])
        user_lng = float(data['user_lng'])
        dest_lat = float(data['dest_lat'])
        dest_lng = float(data['dest_lng'])
        dest_name = data.get('dest_name', 'destination')

        osrm_url = (
            f"http://router.project-osrm.org/route/v1/foot/"
            f"{user_lng},{user_lat};{dest_lng},{dest_lat}"
            f"?steps=true&geometries=geojson&overview=false"
        )
        print(f"Calling OSRM: {osrm_url}")
        route_resp = req.get(osrm_url, timeout=20)
        route_data = route_resp.json()
        print(f"OSRM response code: {route_data.get('code')}")

        if route_data.get('code') != 'Ok':
            return jsonify({'status': 'error', 'message': 'Could not calculate walking route. Please try again.'})

        legs = route_data['routes'][0]['legs'][0]
        raw_steps = legs['steps']
        processed_steps = []

        for step in raw_steps:
            distance_m = step['distance']
            if distance_m < 2:
                continue
            steps_count = max(1, round(distance_m / 0.75))
            maneuver = step.get('maneuver', {})
            mtype = maneuver.get('type', 'continue')
            modifier = maneuver.get('modifier', '')
            name = step.get('name', '').strip()
            location = maneuver.get('location', [None, None])

            if mtype == 'depart':
                instruction = ('Head ' + modifier if modifier else 'Start walking') + (' on ' + name if name else '')
            elif mtype == 'arrive':
                instruction = 'You have arrived at ' + dest_name
            elif mtype == 'turn':
                instruction = 'Turn ' + modifier + (' onto ' + name if name else '')
            elif mtype == 'continue':
                instruction = 'Continue straight' + (' on ' + name if name else '')
            elif mtype == 'new name':
                instruction = 'Continue onto ' + name if name else 'Continue straight'
            elif mtype == 'end of road':
                instruction = 'At the end of the road, turn ' + modifier + (' onto ' + name if name else '')
            elif mtype == 'roundabout':
                exit_num = maneuver.get('exit', 1)
                instruction = f'At the roundabout, take exit {exit_num}' + (' onto ' + name if name else '')
            elif mtype == 'rotary':
                exit_num = maneuver.get('exit', 1)
                instruction = f'At the traffic circle, take exit {exit_num}' + (' onto ' + name if name else '')
            elif mtype == 'fork':
                instruction = 'At the fork, keep ' + modifier + (' on ' + name if name else '')
            elif mtype == 'merge':
                instruction = 'Merge ' + modifier + (' onto ' + name if name else '')
            else:
                instruction = (mtype.replace('-', ' ').capitalize() + ' ' + modifier).strip() + (' on ' + name if name else '')

            processed_steps.append({
                'instruction': instruction,
                'distance_m': round(distance_m),
                'steps_count': steps_count,
                'lat': location[1],
                'lng': location[0],
            })

        total_distance = legs['distance']
        total_steps = round(total_distance / 0.75)
        total_minutes = max(1, round(legs['duration'] / 60))

        print(f"Route found: {len(processed_steps)} steps, {round(total_distance)}m, ~{total_minutes} min")

        return jsonify({
            'status': 'ok',
            'steps': processed_steps,
            'dest_name': dest_name,
            'total_distance': round(total_distance),
            'total_steps': total_steps,
            'total_minutes': total_minutes
        })

    except Exception as e:
        print(f"NAVIGATE ERROR: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return jsonify({'status': 'error', 'message': f'Navigation error: {str(e)}'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True, use_reloader=False)