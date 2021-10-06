import base64

import dash
from dash import html
from dash import dcc
from dash import dash_table as dt
from dash.dependencies import Input, Output, State
import networkx as nx
from networkx.algorithms.flow.capacityscaling import capacity_scaling
import plotly.graph_objects as go
from dataclasses import asdict

from tools.generator import generate_game_state
from tools.file_parser import parse_input_text
from tools.game import GameState, Station, Line, Train, TrainPositionType, PassengerGroup, PassengerGroupPositionType  # pylint: disable=unused-import


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
    max_capacity = max( stations.values(), key=lambda station: station['capacity'])
    max_capacity = max_capacity['capacity']

    sizes=[]
    colors=[]
    opacities=[]
    symbols=[]

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
        elif len(current_trains) == station['capacity']:
            hovertext = f"Trains: {len(current_trains)}"
            symbol="circle"
            color = "red"
        else:
            hovertext = f"Trains: {len(current_trains)}"
            symbol="circle"
            color = "blue"

        hovertext += f"/{station['capacity']}"

        if len(current_passenger_groups) == 0:
            hovertext += "<br>Passengers: 0/0"
            opacity = 0.6
        else:
            passenger_count = 0
            for passenger_group in current_passenger_groups:
                passenger_count += passenger_group['group_size']
            hovertext += f"<br>Passengers: {len(current_passenger_groups)}/{passenger_count}"
            symbol = "diamond"
            opacity = 1

        node_trace['hovertext'] += tuple([hovertext])
        node_trace['customdata'] += tuple(
            [{"trains": [train['name'] for train in current_trains], "passengers": [passenger_group['name'] for passenger_group in current_passenger_groups], "name": station['name']}])
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['text'] += tuple([station['name']])
        colors+=  [color]
        sizes += [station['capacity']]
        symbols += [symbol]
        opacities += [opacity]
    node_trace['marker'] = {
        "color": colors, "size": sizes, "symbol": symbols, "opacity": opacities, "sizemode": 'area', "sizeref": 2.*max_capacity/(20.**2), "sizemin": 5}
    trace_list.append(node_trace)
    return trace_list


def get_edge_traces(pos: dict, lines: dict,
                    trains: dict, passenger_groups: dict):
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
        dummy_current_trains = list(
            filter(
                lambda x,
                id=name: x['position'] == id,
                trains.values()))
        dummy_current_passenger_groups = list(
            filter(
                lambda x, id=name: x['position'] == id,
                passenger_groups.values()
            )
        )
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
            "color": "LightBlue"},
            opacity=1)

        text = name
        middle_hover_trace['text'] += tuple([text])
        middle_hover_trace['x'] += tuple([(x_0 + x_1) / 2])
        middle_hover_trace['y'] += tuple([(y_0 + y_1) / 2])
        trace_list.append(trace)

    trace_list.append(middle_hover_trace)
    return trace_list


def filter_trains_in_station(x):
    return x[1]['position_type'] == TrainPositionType.STATION


def filter_trains_on_line(x):
    return x[1]['position_type'] == TrainPositionType.LINE


def filter_passenger_groups_in_station(x):
    return x[1]['position_type'] == PassengerGroupPositionType.STATION


def filter_passenger_groups_on_line(x):
    return x[1]['position_type'] == PassengerGroupPositionType.TRAIN


def make_plotly_map_from_game_state(game_state_dict: dict, pos: dict):
    plot_traces = []
    plot_traces += get_edge_traces(pos,
                                   game_state_dict['lines'],
                                   dict(filter(filter_trains_on_line,
                                               game_state_dict['trains'].items())),
                                   dict(filter(filter_passenger_groups_on_line,
                                               game_state_dict['passenger_groups'].items())))
    plot_traces += get_node_traces(pos,
                                   game_state_dict['stations'],
                                   dict(filter(filter_trains_in_station,
                                               game_state_dict['trains'].items())),
                                   dict(filter(filter_passenger_groups_in_station,
                                               game_state_dict['passenger_groups'].items())))
    return plot_traces


# Define App and layout
app = dash.Dash(__name__)
app.layout = html.Div(
    children=[
        # Define Storages
        dcc.Store(id='storeGraph', storage_type='session'),
        dcc.Store(id='storeGameState', storage_type='session'),
        # Title div
        html.Div([html.H1("Abfahrt! GUI")],
                 className="row",
                 style={'textAlign': "center"}),
        # Input section
        html.Div(className="row", children=[
            html.H2("Generate new Map"),
            html.Button("Generate", id='generateButton'),
            html.H2("Upload input file"),
            dcc.Upload(
                id='upload-data',
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
            html.H2("Resize Graph"),
            html.Div(children=[
                html.Button("Bigger", id='bigger_button'),
                html.Button("Smaller", id='smaller_button')
            ]),

        ]),
        # Row with visualization
        html.Div(className="row", children=[
            # First column with information about all passengers and trains
            html.Div(className="three columns", style={"height": "100vh", "overflow": "scroll", "position": "sticky", "top": "0"}, children=[
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
            html.Div(className="six columns", children=[
                html.H2("Visualization", style={"text-align": "center"}),
                dcc.Graph(
                    id='map',
                ),
            ]),
            # Third Column with information about passengers and trains of
            # station
            html.Div(className="three columns", style={"height": "100vh", "overflow": "scroll", "position": "sticky", "top": "0"}, children=[
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
    
@ app.callback([Output('map', 'figure'), Output('storeGraph', 'data')],
               [Input('upload-data', 'contents'),
                Input('generateButton', 'n_clicks'),
                Input('smaller_button', 'n_clicks'),
                Input('bigger_button', 'n_clicks'),
                Input('map', 'figure')],
               [State('storeGraph', 'data'), State('storeGameState', 'data')])
def update_output(contents, _, __, ___, figure: go.Figure, traces, game_state):
    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'] == 'upload-data.contents':
        _, content_string = contents.split(',')

        decoded = base64.b64decode(content_string)
        decoded = decoded.decode('utf-8')
        game_state, graph = parse_input_text(decoded)
        pos = get_positions_from_graph(graph)
        game_state_dict = asdict(game_state)
        plot_traces = make_plotly_map_from_game_state(game_state_dict, pos)
    elif ctx.triggered[0]['prop_id'] == 'generateButton.n_clicks':
        game_state, graph = generate_game_state(
            num_stations=200, num_trains=70, num_passengers=60, max_station_capacity=150)
        game_state_dict = asdict(game_state)
        pos = get_positions_from_graph(graph)
        plot_traces = make_plotly_map_from_game_state(game_state_dict, pos)
    elif 'button' in ctx.triggered[0]['prop_id']:
        new_traces = []
        if 'bigger' in ctx.triggered[0]['prop_id']:
            for trace in traces:
                if 'marker' in trace:
                    trace['marker']['size'] = trace['marker']['size'] + 2
                else:
                    trace['line']['width'] = trace['line']['width'] + 1
                new_scatter = go.Scatter(trace)
                new_traces.append(new_scatter)
        else:
            for trace in traces:
                if 'marker' in trace:
                    trace['marker']['size'] = trace['marker']['size'] - 2
                else:
                    trace['line']['width'] = trace['line']['width'] - 1
                new_scatter = go.Scatter(trace)
                new_traces.append(new_scatter)
        figure = go.Figure(data=new_traces, layout=go.Layout(
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
            height=1000)

        )
        return figure, traces
    else:
        game_state = game_state or {}
        plot_traces = traces

    return {
        "data": plot_traces,
        "layout": go.Layout(
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
            height=1000)}, plot_traces


@ app.callback([Output('table_trains', 'filter_query'), Output('table_passengers', 'filter_query')],
               Input('map', 'clickData'))
def update_query(hover_data):
    query = ""
    if hover_data is not None:
        query = '{position} = ' + hover_data['points'][0]['text']
    return query, query


@ app.callback([Output('table_trains', 'data'), Output('table_passengers', 'data')],
               Input('storeGameState', 'data'))
def initialize_tables(game_state):
    sorted_stations = []
    sorted_trains = []
    #if game_state is not None:
        #trains = game_state['trains'].values()
        #stations = game_state['passenger_groups'].values()
        #sorted_trains = sorted(trains, key=lambda train: train['name'])
        #sorted_stations = sorted(stations, key=lambda station: station['name'])
    return sorted_trains, sorted_stations


# Start App
if __name__ == '__main__':
    app.run_server(debug=True)
