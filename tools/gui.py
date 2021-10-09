import base64
from copy import deepcopy
from dataclasses import asdict

import dash
from dash import html
from dash import dcc
from dash import dash_table as dt
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import networkx as nx
import plotly.graph_objects as go

from tools.generator import generate_game_state
from tools.file_parser import parse_input_text, parse_output_text
from abfahrt.types import TrainPositionType, PassengerGroupPositionType  # pylint: disable=unused-import


def get_node_traces(pos: dict, stations: dict,
                    trains: dict, passenger_groups: dict):
    trace_list = []
    node_trace = go.Scatter(
        x=[],
        y=[],
        hovertext=[],
        customdata=[],
        text=[],
        mode='markers+text',
        textposition="bottom center",
        hoverinfo="text",
    )
    max_capacity = max(
        stations.values(),
        key=lambda station: station['capacity'])
    max_capacity = max_capacity['capacity']

    sizes = []
    colors = []
    opacities = []
    symbols = []

    for station in stations.values():
        current_trains = list(
            filter(
                lambda x,
                id=station['name']: x['position'] == id,
                trains.values()))
        current_passenger_groups = list(
            filter(
                lambda x, id=station['name']: x['position'] == id,
                passenger_groups.values()
            )
        )
        x, y = pos[station['name']]
        if len(current_trains) == 0:
            hovertext = "Trains: 0"
            color = "green"
            symbol = "circle-open"
        elif len(current_trains) == int(station['capacity']):
            hovertext = f"Trains: {len(current_trains)}"
            symbol = "circle"
            color = "red"
        else:
            hovertext = f"Trains: {len(current_trains)}"
            symbol = "circle"
            color = "blue"

        hovertext += f"/{station['capacity']}"

        if len(current_passenger_groups) == 0:
            hovertext += "<br>Passengers: 0/0"
            opacity = 0.6
        else:
            passenger_count = 0
            for passenger_group in current_passenger_groups:
                passenger_count += int(passenger_group['group_size'])
            hovertext += f"<br>Passengers: {len(current_passenger_groups)}/{passenger_count}"
            symbol = "diamond"
            opacity = 1

        custom_data = {
            "type": "station",
            "name": station['name'],
            "trains": current_trains}
        node_trace['hovertext'] += tuple([hovertext])
        node_trace['customdata'] += tuple([custom_data])
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['text'] += tuple([station['name']])
        colors += [color]
        sizes += [station['capacity']]
        symbols += [symbol]
        opacities += [opacity]
    node_trace['marker'] = {
        "color": colors, "size": sizes, "symbol": symbols, "opacity": opacities, "sizemode": 'area', "sizeref": 2. * max_capacity / (20.**2), "sizemin": 5}
    trace_list.append(node_trace)
    return trace_list


def get_edge_traces(pos: dict, lines: dict,
                    trains: dict):
    # Traces for lines
    trace_list = []
    # Add point in the middle to make edges hoverable
    middle_hover_trace = go.Scatter(
        x=[],
        y=[],
        hovertext=[],
        text=[],
        customdata=[],
        mode='text',
        hoverinfo="text",
        textposition="bottom center",
        marker={'size': 20, 'opacity': 0},
        opacity=1)
    for line in lines.values():
        name = line['name']
        current_trains = list(
            filter(
                lambda x,
                id=name: x['position'] == id,
                trains.values()))
        if len(current_trains) == 0:
            line_type = "dot"
            color = "LightGreen"
        elif len(current_trains) == line['capacity']:
            color = "red"
            line_type = "solid"
        else:
            color = "yellow"
            line_type = "solid"
        start_point = pos[line['start']]
        x_0 = start_point[0]
        y_0 = start_point[1]
        end_point = pos[line['end']]
        x_1 = end_point[0]
        y_1 = end_point[1]
        trace = go.Scatter(x=tuple([x_0, x_1, None]), y=tuple([y_0, y_1, None]),
                           mode='lines',
                           line={
            'width': line['capacity'] * 3,
            "color": color, "dash": line_type},
            opacity=1)

        text = f"<b>{name}</b><br>Trains</b>: {len(current_trains)}/{line['capacity']}<br>Length: {line['length']}"
        custom_data = {"type": "line", "name": name, "trains": current_trains}
        middle_hover_trace['text'] += tuple([text])
        middle_hover_trace['x'] += tuple([(x_0 + x_1) / 2])
        middle_hover_trace['y'] += tuple([(y_0 + y_1) / 2])
        middle_hover_trace['customdata'] += tuple([custom_data])
        trace_list.append(trace)

    trace_list.append(middle_hover_trace)
    return trace_list


def filter_trains_in_station(x):
    return x[1]['position_type'] == str(TrainPositionType.STATION)


def filter_trains_on_line(x):
    return x[1]['position_type'] == str(TrainPositionType.LINE)


def filter_passenger_groups_in_station(x):
    return x[1]['position_type'] == str(PassengerGroupPositionType.STATION)


def filter_passenger_groups_on_line(x):
    return x[1]['position_type'] == str(PassengerGroupPositionType.TRAIN)


def make_plotly_map_from_game_state(game_state_dict: dict, pos: dict):
    plot_traces = []
    plot_traces += get_edge_traces(pos,
                                   game_state_dict['lines'],
                                   dict(filter(filter_trains_on_line,
                                               game_state_dict['trains'].items())))
    plot_traces += get_node_traces(pos,
                                   game_state_dict['stations'],
                                   dict(filter(filter_trains_in_station,
                                               game_state_dict['trains'].items())),
                                   dict(filter(filter_passenger_groups_in_station,
                                               game_state_dict['passenger_groups'].items())))
    return plot_traces


# Define App and layout
app = dash.Dash(
    __name__,
    title="Abfahrt! GUI",
    update_title="Abfahrt! GUI, updating...")
app.layout = html.Div(
    children=[
        # Define Storages
        dcc.Store(id='store_current_game_state', storage_type='session'),
        dcc.Store(id='store_game_states', storage_type='session'),
        # Title div
        html.Div([html.H1("Abfahrt! GUI")],
                 className="",
                 style={'textAlign': "center"}),
        # Input section
        html.Div(id="input_section", className="row", children=[
            html.Div(children=[
                html.H2("Generate new Map"),
                html.Button("Generate", id='generate_button'),
            ]),
            html.Div(id="file_inputs", children=[
                html.H2("Upload input file"),
                dcc.Upload(
                    id='upload-input',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px auto'
                    },
                    accept="text/plain",
                ),
                html.H2("Upload output File"),
                dcc.Upload(
                    id='upload-output',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px auto'
                    },
                    accept="text/plain",
                ),
            ]),
            html.Div(
                children=[
                    html.H2("Select Round"),
                    dcc.RadioItems(
                        id="round_dropdown",

                    ),
                ]
            ),
        ]),
        # Row with visualization
        html.Div(className="row", children=[
            # First column with information about all passengers and trains
            html.Div(className="data_tables", style={}, children=[
                html.Div(style={}, children=[
                    html.H2("Trains"),
                    dt.DataTable(id="table_trains", columns=[
                        {'name': "Name", "id": "name"},
                        {'name': "Position", "id": "position"},
                        {"name": "Capacity", "id": "capacity", "hideable": True},
                        {"name": "speed", "id": "speed", "hideable": True},
                        {"name": "Passengers", "id": "passengers", "hideable": True}
                    ], filter_action="native",
                        sort_action="native",
                        sort_mode="multi",
                    )
                ]),
            ]),
            # Second column
            html.Div(className="middle_column", children=[
                html.Div(className="row", style={"justify-content": "space-between"}, children=[
                    html.Button(
                        id="round_decrement",
                        children=["Previous Round"],
                        n_clicks=0),
                    html.H2("Visualization", style={"text-align": "center"}),
                    html.Button(
                        id="round_increment",
                        children=["Next Round"],
                        n_clicks=0
                    ),
                ]),
                dcc.Graph(
                    id='map',
                    config={
                        'toImageButtonOptions': {
                            'format': 'svg',
                            'filename': 'network_graph',
                            'height': 1920,
                            'width': 1080,
                            'scale': 1
                        },
                        "scrollZoom": True,
                        "displaylogo": False,
                        "modeBarButtons": [
                            ["pan2d", "zoom2d", "zoomIn2d",
                                "zoomOut2d", "autoScale2d", "toImage"]
                        ]},
                    figure={"layout": go.Layout(
                        title='Map full of stations and train lines',
                        showlegend=False,
                        hovermode='closest',
                        margin={
                            'b': 40,
                            'l': 40,
                            'r': 40,
                            't': 40},
                        xaxis={
                            'showgrid': False,
                            'zeroline': False,
                            'showticklabels': False},
                        yaxis={
                            'showgrid': False,
                            'zeroline': False,
                            'showticklabels': False},
                        height=1000)}),
            ]),
            # Third Column with information about passengers and trains of
            # station
            html.Div(className="data_tables", style={}, children=[
                html.Div(style={}, children=[
                    html.H2("Passengers"),
                    dt.DataTable(id="table_passengers", columns=[
                        {'name': "Name", "id": "name"},
                        {'name': "Position", "id": "position"},
                        {"name": "Size", "id": "group_size", "hideable": True},
                        {'name': "Time", "id": 'time_remaining', "hideable": True}
                    ], filter_action="native",
                        sort_action="native",
                        sort_mode="multi",
                    )
                ],
                )
            ]
            )
        ]),
    ])


### Callbacks ###

# Callback to handle input file
def get_positions_from_graph(graph: nx.Graph):
    pos = nx.kamada_kawai_layout(graph)
    new_pos = {name: tuple(arr) for name, arr in pos.items()}
    return new_pos


def make_dict_serializable(game_state_dict: dict):
    serializable_game_state_dict = {}
    trains = {}
    for train in game_state_dict['trains'].values():
        train['position_type'] = str(train['position_type'])
        trains[train['name']] = train
    passenger_groups = {}
    for passenger_group in game_state_dict['passenger_groups'].values():
        passenger_group['position_type'] = str(
            passenger_group['position_type'])
        passenger_groups[passenger_group['name']] = passenger_group
    serializable_game_state_dict['trains'] = trains
    serializable_game_state_dict['passenger_groups'] = passenger_groups
    serializable_game_state_dict['stations'] = game_state_dict['stations']
    serializable_game_state_dict['lines'] = game_state_dict['lines']
    return serializable_game_state_dict


@ app.callback([Output('store_game_states', 'data')],
               [Input('upload-input', 'contents'),
                Input('upload-output', 'contents'),
                Input('generate_button', 'n_clicks'), ])
def update_output(contents, output_content, _,):
    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'] == 'upload-input.contents':
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        decoded = decoded.decode('utf-8')
        game_state, graph = parse_input_text(decoded)
        pos = get_positions_from_graph(graph)
        game_state_dict = asdict(game_state)
        game_state_dict = make_dict_serializable(game_state_dict)
        return [{"positions": pos, "game_states": {0: game_state_dict}}]

    elif ctx.triggered[0]['prop_id'] == 'upload-output.contents' and contents is not None:
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        decoded = decoded.decode('utf-8')
        _, output_content_string = output_content.split(',')
        decoded_output = base64.b64decode(output_content_string)
        decoded_output = decoded_output.decode('utf-8')
        game_state, graph = parse_input_text(decoded)
        pos = get_positions_from_graph(graph=graph)
        schedule = parse_output_text(decoded_output)
        num_rounds = max(schedule.actions.keys())
        start_round = 0 if schedule.actions[0].is_zero_round() else 1
        game_state_dicts = {}
        for i in range(start_round, num_rounds + 1):
            action = schedule.actions[i]
            new_game_state = deepcopy(game_state)
            new_game_state.apply(action)
            game_state_dicts[i] = make_dict_serializable(asdict(game_state))
            game_state = new_game_state

        return [{"positions": pos, "game_states": game_state_dicts}]
    elif ctx.triggered[0]['prop_id'] == 'generate_button.n_clicks':
        game_state, graph = generate_game_state(num_stations=100,
                                                num_trains=100, num_passengers=100)
        pos = get_positions_from_graph(graph)
        return [{"positions": pos, "game_states": {
            0: make_dict_serializable(asdict(game_state))}}]
    raise PreventUpdate


@app.callback(Output('round_dropdown', 'value'),
              [Input('store_game_states', 'data'),
               Input('round_dropdown', 'value'),
               Input('round_increment', 'n_clicks'),
               Input('round_decrement', 'n_clicks')])
def select_round(all_game_states, old_index, n_clicks, n_clicks_2):  # pylint: disable=unused-argument
    if all_game_states is None:
        raise PreventUpdate
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    index = ''
    if trigger_id == 'round_increment':
        if not old_index:
            raise PreventUpdate
        index = str(int(old_index) + 1)
    elif trigger_id == 'round_decrement':
        if not old_index:
            raise PreventUpdate
        index = str(int(old_index) - 1)
    elif len(all_game_states['game_states']) != 0:
        index = '0'
    if index != '' and index not in all_game_states['game_states']:
        raise PreventUpdate
    return index


@ app.callback(Output('round_dropdown', "options"),
               Input('store_game_states', 'data'))
def set_options(all_game_states):
    if all_game_states is None:
        raise PreventUpdate
    game_states = all_game_states['game_states']
    options = []
    for i in game_states.keys():
        options.append({"label": f"Round {i}", "value": i})
    return options


@ app.callback(Output('store_current_game_state', 'data'),
               Input('round_dropdown', 'value'), State('store_game_states', 'data'))
def update_current_game_state(index, all_game_states):
    if index == '' or index is None or all_game_states is None or all_game_states == {}:
        raise PreventUpdate
    return {"positions": all_game_states["positions"],
            "game_state": all_game_states['game_states'][index] or {}}


@ app.callback([Output('table_trains', 'filter_query'), Output('table_passengers', 'filter_query')],
               Input('map', 'clickData'))
def update_query(hover_data):
    query = ""
    if hover_data is not None:
        data = hover_data['points'][0]['customdata']

        query_list = []
        query_list.append('{position} = ' + data['name'])
        for train in data['trains']:
            query_list.append("{position} = " +
                              train['name'])

        query = " || ".join(query_list)

    if query == "":
        raise PreventUpdate
    return query, query


@ app.callback([Output('table_trains', 'data'), Output('table_passengers', 'data'), Output('map', 'figure')],
               Input('store_current_game_state', 'data'))
def initialize_tables(game_information):
    if game_information == [{}] or game_information is None:
        raise PreventUpdate
    game_state = game_information['game_state']
    positions = game_information['positions']
    plot_traces = make_plotly_map_from_game_state(game_state, positions)
    sorted_stations = []
    sorted_trains = []
    if game_state is not None:
        trains = game_state['trains'].values()
        stations = game_state['passenger_groups'].values()
        sorted_trains = sorted(trains, key=lambda train: train['name'])
        sorted_stations = sorted(stations, key=lambda station: station['name'])
    return sorted_trains, sorted_stations, {
        "data": plot_traces,
        "layout": go.Layout(
            title='Map full of stations and train lines',
            showlegend=False,
            autosize=True,
            hovermode='closest',
            margin={
                'b': 40,
                'l': 40,
                'r': 40,
                't': 40},
            xaxis={
                'showgrid': False,
                'zeroline': False,
                'showticklabels': False},
            yaxis={
                'showgrid': False,
                'zeroline': False,
                'showticklabels': False},
        )}


# Start App
if __name__ == '__main__':
    app.run_server(debug=True)
