import sys
import json


def label_studio_json_to_dict(json_content):
    """Cleans the label studio JSON file to a much more readable format of labels"""
    result = {}
    for image_label in json_content:
        label_content = {'figures': []}
        filename_dash_pos = str(image_label['file_upload']).find('-')
        filename = str(image_label['file_upload'])[filename_dash_pos+1:]

        annotation = image_label['annotations'][0]['result']
        for label in annotation:
            if label['type'] == 'keypointlabels':
                type = label['value']['keypointlabels'][0]
                px = float(label['value']['x'])/100
                py = float(label['value']['y'])/100
                label_content['figures'].append({'type': type, 'x': px, 'y': py})
            elif label['type'] == 'polygonlabels':
                type = label['value']['polygonlabels'][0]
                points = []
                for pt in label['value']['points']:
                    px = float(pt[0])/100
                    py = float(pt[1])/100
                    points.append({'x': px, 'y': py})
                label_content[type] = points

        result[filename] = label_content

    return result


def print_usage():
    print("Usage: python ls_json_to_dict <json path> <output path>")


if __name__ == '__main__':
    try:
        in_name = sys.argv[1]
        out_name = sys.argv[2]
    except:
        print_usage()
        exit()

    with open(in_name, 'r') as json_file:
        json_content = json.load(json_file)

    result = label_studio_json_to_dict(json_content)
    json_out = json.dumps(result, sort_keys=True, indent=4)

    with open(out_name, 'w') as out_f:
        out_f.write(json_out)
