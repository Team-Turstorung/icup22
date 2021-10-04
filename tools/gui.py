import base64
from file_parser import parse_text
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import networkx as nx
import plotly.graph_objs as go


def get_node_traces(world: nx.Graph):
    trace_list = []
    for node in world.nodes:
        node_trace = go.Scatter(
            x=[],
            y=[],
            hovertext=[],
            text=[],
            mode='markers+text',
            textposition="bottom center",
            hoverinfo="text")
        x, y = world.nodes[node]['pos']
        capacity = world.nodes[node]["capacity"]
        hovertext = f"Kapazität: {capacity}"
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['hovertext'] += tuple([hovertext])
        node_trace['text'] += tuple([node])
        node_trace['marker'] = {
            "size": 5 + world.nodes[node]["capacity"] * 5,
            "color": "LightBlue"}
        trace_list.append(node_trace)
    return trace_list


def get_edge_traces(world: nx.Graph):
    trace_list = []
    # Traces for line
    for edge in world.edges:
        x_0, y_0 = world.nodes[edge[0]]['pos']
        x_1, y_1 = world.nodes[edge[1]]['pos']
        trace = go.Scatter(x=tuple([x_0, x_1, None]), y=tuple([y_0, y_1, None]),
                           mode='lines',
                           opacity=1)
        trace_list.append(trace)

    # Add point in the middle to make edges hoverable
    middle_hover_trace = go.Scatter(
        x=[],
        y=[],
        hovertext=[],
        mode='markers',
        hoverinfo="text",
        marker={
            'size': 20,
            'color': 'LightSkyBlue'},
        opacity=0)
    for edge in world.edges:
        x_0, y_0 = world.nodes[edge[0]]['pos']
        x_1, y_1 = world.nodes[edge[1]]['pos']
        hovertext = "From: " + edge[0] + "<br>" + "To: " + edge[1] + "<br>" + "capacity: " + str(
            world.edges[edge]['capacity']) + "<br>" + "length: " + str(world.edges[edge]['length'])
        middle_hover_trace['x'] += tuple([(x_0 + x_1) / 2])
        middle_hover_trace['y'] += tuple([(y_0 + y_1) / 2])
        middle_hover_trace['hovertext'] += tuple([hovertext])
    trace_list.append(middle_hover_trace)
    return trace_list


def make_plotly_map(world: nx.Graph):
    pos = nx.layout.kamada_kawai_layout(world, weight="length")
    for node in world.nodes:
        world.nodes[node]['pos'] = list(pos[node])

    plot_traces = []
    plot_traces += get_node_traces(world)
    plot_traces += get_edge_traces(world)
    return plot_traces


# Define App and layout
app = dash.Dash(__name__)
app.layout = html.Div(className="row",
                      children=[
                          html.H1(children='Abfahrt! GUI'),

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
                                  'margin': '10px'
                              },
                              accept="text/plain",
                          ),
                          html.H2("Visualization"),
                          dcc.Graph(
                              id='map',
                          ),
                      ])

### Callbacks ###

# Callback to handle input file


@app.callback(Output('map', 'figure'),
              Input('upload-data', 'contents'))
def update_output(contents):
    if contents is not None:
        _, content_string = contents.split(',')

        decoded = base64.b64decode(content_string)
        decoded = decoded.decode('utf-8')
    else:
        decoded = ""

    world = parse_text(decoded)
    plot_traces = make_plotly_map(world=world)
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
            height=1000)}


# Start App
if __name__ == '__main__':
    app.run_server(debug=True)
