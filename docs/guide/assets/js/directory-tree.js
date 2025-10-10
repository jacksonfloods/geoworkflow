/**
 * Organization Chart Style Directory Tree using D3.js v7
 * Vertical layout with expand/collapse buttons on the right
 * 
 * Features:
 * - Vertical tree layout (top-to-bottom)
 * - Modern card-based layout
 * - Smooth expand/collapse animations
 * - Professional styling
 * - Hover effects with tooltips
 * - Click to expand/collapse subtrees
 */

class OrgChart {
  constructor() {
    this.data = null;
    this.svg = null;
    this.chart = null;
    this.attrs = {
      id: 'tree-container',
      svgWidth: 1200,
      svgHeight: 1200,
      marginTop: 20,
      marginBottom: 20,
      marginRight: 20,
      marginLeft: 20,
      duration: 400,
      nodeWidth: 240,
      nodeHeight: 100,
      childrenMargin: 220,         // Vertical space between levels (reduced)
      compactMarginPair: 80,
      compactMarginBetween: 1,   // Horizontal space between sibling nodes (reduced)
      onNodeClick: null,
      nodeContent: null,
      nodeUpdate: null,
      nodeEnter: null,
      initialZoom: 1,
      backgroundColor: '#fafafa'
    };
  }

  async init() {
    try {
      // Fetch data
      const response = await fetch('../assets/directory-tree.json');
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      this.data = await response.json();

      // Initialize chart
      this.initializeChart();
      this.update(this.data);

      // Hide loading
      document.getElementById('loading').style.display = 'none';
    } catch (error) {
      this.showError(error.message);
    }
  }

  initializeChart() {
    const container = d3.select(`#${this.attrs.id}`);
    container.selectAll('*').remove();

    // Create SVG
    this.svg = container
      .append('svg')
      .attr('width', '100%')
      .attr('height', this.attrs.svgHeight)
      .attr('font-family', 'Roboto, sans-serif');

    // Create chart group
    this.chart = this.svg
      .append('g')
      .attr('transform', `translate(${this.attrs.marginLeft}, ${this.attrs.marginTop})`);

    // Add zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.1, 3])
      .on('zoom', (event) => {
        this.chart.attr('transform', event.transform);
      });

    this.svg.call(zoom);

    // Set initial zoom and position (center horizontally, near top)
    this.svg.call(zoom.transform, d3.zoomIdentity.translate(
      this.attrs.svgWidth / 2,
      100
    ).scale(this.attrs.initialZoom));
  }

  update(source) {
    const attrs = this.attrs;

    // Compute the layout
    const root = d3.hierarchy(source);

    // Store children references
    root.descendants().forEach((d, i) => {
      d.id = i;
      d._children = d.children;
      if (d.depth && d.data.children) {
        d.children = null; // Start collapsed
      }
    });

    this.root = root;
    this.root.x0 = 0;
    this.root.y0 = 0;

    this.updateTree(this.root);
  }

  updateTree(source) {
    const attrs = this.attrs;

    // Create tree layout
    const treeLayout = d3.tree()
      .nodeSize([attrs.nodeWidth + attrs.compactMarginBetween,
      attrs.nodeHeight + attrs.childrenMargin])
      .separation((a, b) => a.parent === b.parent ? 1 : 1.2);

    // Compute the layout
    treeLayout(this.root);

    const nodes = this.root.descendants();
    const links = this.root.links();

    // Swap x and y coordinates for vertical orientation
    nodes.forEach(d => {
      const temp = d.x;
      d.x = d.y;
      d.y = temp;
    });

    // Transition setup
    const transition = this.svg.transition()
      .duration(attrs.duration)
      .tween('resize', () => () => this.svg.dispatch('toggle'));

    // *************** LINKS (RENDER FIRST - BEHIND NODES) ***************
    const link = this.chart
      .selectAll('.link')
      .data(links, d => d.target.id);

    // Enter new links
    const linkEnter = link.enter()
      .append('path')
      .attr('class', 'link')
      .attr('d', d => {
        const o = { x: source.x0, y: source.y0 };
        return this.diagonal(o, o);
      })
      .style('fill', 'none')
      .style('stroke', '#ccc')
      .style('stroke-width', '2px')
      .style('opacity', 0);

    // UPDATE links
    const linkUpdate = linkEnter.merge(link);

    linkUpdate.transition(transition)
      .attr('d', d => this.diagonal(d.source, d.target))
      .style('opacity', 1);

    // EXIT old links
    link.exit()
      .transition(transition)
      .attr('d', d => {
        const o = { x: source.x, y: source.y };
        return this.diagonal(o, o);
      })
      .style('opacity', 0)
      .remove();

    // *************** NODES (RENDER SECOND - ON TOP) ***************
    const node = this.chart
      .selectAll('.node')
      .data(nodes, d => d.id);

    // Enter new nodes
    const nodeEnter = node.enter()
      .append('g')
      .attr('class', 'node')
      .attr('transform', d => `translate(${source.x0},${source.y0})`)
      .style('cursor', 'pointer')
      .on('click', (event, d) => this.onNodeClick(event, d));

    // Add card background
    nodeEnter.append('rect')
      .attr('class', 'node-rect')
      .attr('width', attrs.nodeWidth)
      .attr('height', attrs.nodeHeight)
      .attr('x', -attrs.nodeWidth / 2)
      .attr('y', -attrs.nodeHeight / 2)
      .attr('rx', 8)
      .attr('ry', 8)
      .style('fill', d => this.getNodeColor(d))
      .style('stroke', d => this.getNodeBorderColor(d))
      .style('stroke-width', '2px')
      .style('filter', 'drop-shadow(0 2px 8px rgba(0,0,0,0.1))')
      .style('opacity', 0);

    // Add icon
    nodeEnter.append('text')
      .attr('class', 'node-icon')
      .attr('text-anchor', 'middle')
      .attr('y', -20)
      .style('font-size', '48px')
      .text(d => this.getNodeIcon(d))
      .style('opacity', 0);

    // Add name text
    nodeEnter.append('text')
      .attr('class', 'node-name')
      .attr('text-anchor', 'middle')
      .attr('y', 10)
      .style('font-size', '14px')
      .style('font-weight', '600')
      .style('fill', '#333')
      .text(d => this.truncateText(d.data.name, 30))
      .style('opacity', 0);

    // Add type badge
    nodeEnter.append('text')
      .attr('class', 'node-type')
      .attr('text-anchor', 'middle')
      .attr('y', 28)
      .style('font-size', '11px')
      .style('fill', '#666')
      .text(d => d.data.type === 'directory' ? '📁 Directory' : '📄 File')
      .style('opacity', 0);

    // Add expand/collapse indicator (circle)
    nodeEnter.append('circle')
      .attr('class', 'node-expand')
      .attr('cx', attrs.nodeWidth / 2 + 20)  // To the right of the node
      .attr('cy', 0)                          // Vertically centered
      .attr('r', 12)
      .style('fill', '#fff')
      .style('stroke', '#2196F3')
      .style('stroke-width', '2px')
      .style('opacity', d => d._children ? 1 : 0);

    // Add expand/collapse indicator (text)
    nodeEnter.append('text')
      .attr('class', 'node-expand-text')
      .attr('text-anchor', 'middle')
      .attr('x', attrs.nodeWidth / 2 + 20)  // To the right of the node
      .attr('y', 5)                          // Vertically centered (slight offset for baseline)
      .style('font-size', '14px')
      .style('font-weight', 'bold')
      .style('fill', '#2196F3')
      .style('pointer-events', 'none')
      .text(d => d._children ? '+' : '')
      .style('opacity', d => d._children ? 1 : 0);

    // UPDATE existing nodes
    const nodeUpdate = nodeEnter.merge(node);

    nodeUpdate.transition(transition)
      .attr('transform', d => `translate(${d.x},${d.y})`);

    nodeUpdate.select('.node-rect')
      .transition(transition)
      .style('fill', d => this.getNodeColor(d))
      .style('opacity', 1);

    nodeUpdate.select('.node-icon')
      .transition(transition)
      .style('opacity', 1);

    nodeUpdate.select('.node-name')
      .transition(transition)
      .style('opacity', 1);

    nodeUpdate.select('.node-type')
      .transition(transition)
      .style('opacity', 1);

    nodeUpdate.select('.node-expand')
      .transition(transition)
      .style('opacity', d => d._children || d.children ? 1 : 0);

    nodeUpdate.select('.node-expand-text')
      .text(d => d.children ? '−' : '+')
      .transition(transition)
      .style('opacity', d => d._children || d.children ? 1 : 0);

    // EXIT old nodes
    const nodeExit = node.exit()
      .transition(transition)
      .attr('transform', d => `translate(${source.x},${source.y})`)
      .remove();

    nodeExit.select('.node-rect')
      .style('opacity', 0);

    nodeExit.selectAll('text')
      .style('opacity', 0);

    // Store old positions for next transition
    nodes.forEach(d => {
      d.x0 = d.x;
      d.y0 = d.y;
    });

    // Add tooltips
    this.addTooltips(nodeUpdate);
  }

  onNodeClick(event, d) {
    if (d.children) {
      d._children = d.children;
      d.children = null;
    } else if (d._children) {
      d.children = d._children;
      d._children = null;
    }
    this.updateTree(d);
  }

  diagonal(s, d) {
    const attrs = this.attrs;

    // Start from bottom center of parent
    const startX = s.x;
    const startY = s.y + attrs.nodeHeight / 2;

    // End at top center of child
    const endX = d.x;
    const endY = d.y - attrs.nodeHeight / 2;

    // Create smooth curved path
    const midY = (startY + endY) / 2;

    const path = `M ${startX} ${startY}
                  C ${startX} ${midY},
                    ${endX} ${midY},
                    ${endX} ${endY}`;
    return path;
  }

  getNodeColor(d) {
    if (d.data.type === 'directory') {
      return d.depth === 0 ? '#673AB7' : '#9575CD'; // Deep purple shades
    }
    return '#E8F5E9'; // Light green for files
  }

  getNodeBorderColor(d) {
    if (d.data.type === 'directory') {
      return d.depth === 0 ? '#512DA8' : '#7E57C2';
    }
    return '#4CAF50';
  }

  getNodeIcon(d) {
    if (d.data.type === 'directory') {
      return d.children ? '📂' : '📁';
    }
    return '📄';
  }

  truncateText(text, maxLength) {
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
  }

  addTooltips(nodeUpdate) {
    nodeUpdate
      .on('mouseover', (event, d) => {
        if (d.data.description) {
          const tooltip = d3.select('#tooltip');
          tooltip
            .style('opacity', 0.95)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px')
            .html(`
              <div style="font-weight: bold; margin-bottom: 8px; font-size: 13px;">
                ${d.data.name}
              </div>
              <div style="color: #666; font-size: 11px; margin-bottom: 8px;">
                ${d.data.type === 'directory' ? '📁 Directory' : '📄 File'}
              </div>
              <div style="padding-top: 8px; border-top: 1px solid #ddd; font-size: 12px;">
                ${d.data.description}
              </div>
            `);
        }
      })
      .on('mouseout', () => {
        d3.select('#tooltip').style('opacity', 0);
      });
  }

  showError(message) {
    document.getElementById('loading').style.display = 'none';
    const container = d3.select(`#${this.attrs.id}`);
    container.html(`
      <div style="padding: 40px; text-align: center; color: #d32f2f;">
        <h3>Error Loading Directory Tree</h3>
        <p>${message}</p>
      </div>
    `);
  }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    const chart = new OrgChart();
    chart.init();
  });
} else {
  const chart = new OrgChart();
  chart.init();
}