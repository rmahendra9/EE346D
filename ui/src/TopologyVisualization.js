// TopologyVisualization.js
// TopologyVisualization.js
import React, { useEffect } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';
import cytoscape from 'cytoscape';
import dagre from 'cytoscape-dagre'

cytoscape.use(dagre)

const TopologyVisualization = ({ adjacencyList }) => {
  useEffect(() => {
    // Create a new Cytoscape instance
    const cy = cytoscape({
      container: document.getElementById('cy'),
      elements: {
        nodes: Object.keys(adjacencyList).map(id => ({ data: { id } })),
        edges: Object.entries(adjacencyList).map(([source, targets]) =>
          targets.map(target => ({ data: { source, target } }))
        ).flat()
      },
      layout: {
        name: 'dagre', // Use the DAG layout algorithm
        center: true
      },
      style: [
        {
          selector: 'node',
          style: {
            'background-color': '#666',
            'label': 'data(id)'
          }
        },
        {
          selector: 'edge',
          style: {
            'width': 3,
            'line-color': '#ccc',
            'source-arrow-color': '#ccc',
            'source-arrow-shape': 'triangle',
            'curve-style': 'bezier' // Use bezier curves for edges
          }
        }
      ],
      userZoomingEnabled: false
    });

    // Optionally, add event listeners or other customizations here


    return () => {
      // Cleanup when the component unmounts
      cy.destroy();
    };
  }, [adjacencyList]);

  return <CytoscapeComponent id="cy" style={{display: 'flex', width: '100%', height: '500px', alignContent: "flex-start"}} />;
};

export default TopologyVisualization;
