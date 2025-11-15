// ====== State & Elements ======
const screens = {
  upload: document.getElementById('screen-upload'),
  loading: document.getElementById('screen-loading'),
  results: document.getElementById('screen-results')
}
const fileInput = document.getElementById('file');
const fileName = document.getElementById('file-name');
const fileError = document.getElementById('file-error');
const classifyBtn = document.getElementById('btn-classificar');
const modelSelect = document.getElementById('model');

// Model status elements
const modelStatus = {
  'bert-emotions': document.getElementById('status-bert-emotions'),
  'bert-sentiment': document.getElementById('status-bert-sentiment'),
  'mnb': document.getElementById('status-mnb')
};

// Model info message element
const modelInfo = document.getElementById('model-info');

// Global variable to store parsed CSV data
let csvData = [];

const modal = document.getElementById('modal');
const downloadBtn = document.getElementById('btn-download');
document.getElementById('btn-metrics').onclick = async () => {
  await loadModelMetrics();
  modal.classList.add('open');
};
document.getElementById('close-modal').onclick = () => modal.classList.remove('open');
modal.addEventListener('click', (e)=>{ if(e.target===modal) modal.classList.remove('open'); })
if (downloadBtn) {
  downloadBtn.addEventListener('click', downloadResults);
  downloadBtn.disabled = true;
}

// Load model metrics from API
async function loadModelMetrics() {
  const selectedModel = modelSelect.value;
  const modelNameMap = {
    'bert-emotions': 'BERTimbau Emoções',
    'bert-sentiment': 'BERTimbau Sentimentos',
    'mnb': 'MNB Sentimentos'
  };
  
  // Update model name
  document.getElementById('metric-model').textContent = modelNameMap[selectedModel] || 'Modelo';
  
  try {
    const response = await fetch(`http://localhost:5000/metrics?model=${selectedModel}`);
    if (!response.ok) {
      throw new Error('Failed to load metrics');
    }
    
    const metrics = await response.json();
    
    // Update metrics display
    document.getElementById('acc').textContent = (metrics.accuracy * 100).toFixed(1) + '%';
    document.getElementById('f1').textContent = metrics.f1_macro.toFixed(3);
    document.getElementById('prec').textContent = metrics.precision.toFixed(3);
    document.getElementById('recall').textContent = metrics.recall.toFixed(3);
    document.getElementById('metric-description').textContent = metrics.description || '';
  } catch (error) {
    console.error('Error loading metrics:', error);
    // Fallback to default values
    document.getElementById('acc').textContent = '-';
    document.getElementById('f1').textContent = '-';
    document.getElementById('prec').textContent = '-';
    document.getElementById('recall').textContent = '-';
    document.getElementById('metric-description').textContent = 'Erro ao carregar métricas';
  }
}

function downloadResults() {
  if (!csvData || csvData.length === 0) return;
  const selectedModel = modelSelect.value || 'modelo';
  const header = ['text', 'prediction'];
  const rows = csvData.map(row => {
    const prediction = row.feeling && row.feeling !== 'NULL' && row.feeling !== 'NONE'
      ? row.feeling
      : row.emotion || '';
    const fields = [
      row.text ?? '',
      prediction
    ];
    return fields
      .map(value => `"${String(value).replace(/"/g, '""')}"`)
      .join(';');
  });

  const csvContent = [header.join(';'), ...rows].join('\n');
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const now = new Date();
  const timestamp = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}_${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}${String(now.getSeconds()).padStart(2, '0')}`;
  const filename = `resultados_${selectedModel}_${timestamp}.csv`;

  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(link.href);
}

// File picker is handled by HTML 'for' attribute on the label

// File validation
fileInput.addEventListener('change', () => {
  const f = fileInput.files[0];
  if(!f){ fileName.textContent = 'Nenhum arquivo selecionado'; return; }
  fileName.textContent = f.name;
  const lower = f.name.toLowerCase();
  const valid = lower.endsWith('.csv') || lower.endsWith('.xlsx') || lower.endsWith('.xls');
  fileError.style.display = valid ? 'none' : 'block';
  if(!valid) fileName.textContent = 'Nenhum arquivo selecionado';
});

// Navigation helpers
function show(id){ Object.values(screens).forEach(el=>el.classList.remove('active')); screens[id].classList.add('active'); }

// ====== CSV Parsing Functions ======
function parseCSV(csvText) {
  const lines = csvText.split('\n');
  const headers = lines[0].split(';');
  const data = [];
  
  for (let i = 1; i < lines.length; i++) {
    if (lines[i].trim()) {
      const values = parseCSVLine(lines[i]);
      if (values.length >= 3) {
        data.push({
          text: values[0] || '',
          emotion: values[1] || '',
          feeling: values[2] || ''
        });
      }
    }
  }
  
  return data;
}

function parseCSVLine(line) {
  const values = [];
  let current = '';
  let inQuotes = false;
  
  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    
    if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === ';' && !inQuotes) {
      values.push(current.trim());
      current = '';
    } else {
      current += char;
    }
  }
  
  values.push(current.trim());
  return values;
}

function processEmotionData(data) {
  const emotionCounts = {};
  const feelingCounts = { Positivo: 0, Negativo: 0, Neutro: 0 };
  
  data.forEach(row => {
    // Process emotions (can be multiple separated by semicolons)
    if (row.emotion && row.emotion.trim()) {
      const emotions = row.emotion.split(';').map(e => e.trim().toUpperCase());
      emotions.forEach(emotion => {
        if (emotion && emotion !== '') {
          // Normalize emotion names
          const normalizedEmotion = normalizeEmotionName(emotion);
          emotionCounts[normalizedEmotion] = (emotionCounts[normalizedEmotion] || 0) + 1;
        }
      });
    }
    
    // Process feelings
    if (row.feeling && row.feeling.trim()) {
      const feeling = row.feeling.trim().toUpperCase();
      if (feeling === 'POSITIVO') feelingCounts.Positivo++;
      else if (feeling === 'NEGATIVO') feelingCounts.Negativo++;
      else if (feeling === 'NEUTRO') feelingCounts.Neutro++;
    }
  });
  
  return { emotionCounts, feelingCounts };
}

function normalizeEmotionName(emotion) {
  const emotionMap = {
    'RAIVA': 'Raiva',
    'ALEGRIA': 'Alegria', 
    'AMOR': 'Amor',
    'MEDO': 'Medo',
    'TRISTEZA': 'Tristeza',
    'CONFIANÇA': 'Confiança',
    'CONFIANCA': 'Confiança',
    'AUSENTE': 'Ausente'
  };
  
  return emotionMap[emotion] || emotion;
}

classifyBtn.addEventListener('click', async () => {
  const file = fileInput.files[0];
  if (!file || (!file.name.toLowerCase().endsWith('.csv') && !file.name.toLowerCase().endsWith('.xlsx') && !file.name.toLowerCase().endsWith('.xls'))) {
    fileError.style.display = 'block';
    return;
  }
  
  // Get selected model from dropdown
  const selectedValue = modelSelect.value;
  const selectedModels = [selectedValue];
  
  try {
    // Show loading screen
    show('loading');
    
    // Show status for selected models
    Object.keys(modelStatus).forEach(key => {
      modelStatus[key].style.display = selectedModels.includes(key) ? 'flex' : 'none';
    });
    
    // Read CSV file
    await callSingleModelAPI(file, selectedModels[0]);
    
  } catch (error) {
    console.error('Error during classification:', error);
    fileError.style.display = 'block';
    fileError.textContent = '*Erro ao processar o arquivo. Verifique se a API está rodando.';
    show('upload');
  }
});

// API calling functions
async function callSingleModelAPI(file, modelType = 'bert-emotions') {
  const url = `http://localhost:5000/predict_csv?model=${modelType}`;
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(url, {
    method: 'POST',
    body: formData
  });
  
  if (!response.ok) {
    throw new Error(`API Error: ${response.status}`);
  }
  
  const apiResult = await response.json();
  console.log(`${modelType.toUpperCase()} API result:`, apiResult);
  
  // Convert API result to our format
  csvData = apiResult.results.map(item => ({
    text: item.text,
    emotion: item.emotion,
    feeling: item.feeling,
    confidence: typeof item.confidence === 'number' ? item.confidence : null
  }));
  
  console.log('Processed data for charts:', csvData);
  
  // Show results
  show('results');
  drawAllCharts(csvData);
  if (downloadBtn) {
    downloadBtn.disabled = csvData.length === 0;
  }
}

// ====== Minimal chart drawing (no external libs) ======
function drawPie(canvasId, segments){
  const c = document.getElementById(canvasId);
  const ctx = c.getContext('2d');
  const w = c.width = c.offsetWidth * devicePixelRatio;
  const h = c.height = c.offsetHeight * devicePixelRatio;
  ctx.clearRect(0,0,w,h);
  const cx = w/2, cy = h/2, r = Math.min(w,h)*0.35;
  const total = segments.reduce((a,b)=>a+b.value,0);
  let a0 = -Math.PI/2;
  
  // Custom colors for sentiment labels
  const getSentimentColor = (label) => {
    if (label === 'Negativo') return '#dc2626'; // Red
    if (label === 'Positivo') return '#16a34a'; // Green
    return '#6b7280'; // Default gray
  };
  
  segments.forEach((s,i)=>{
    const a1 = a0 + 2*Math.PI*(s.value/total);
    ctx.beginPath(); ctx.moveTo(cx,cy); ctx.arc(cx,cy,r,a0,a1); ctx.closePath();
    ctx.fillStyle = getSentimentColor(s.label); ctx.fill(); a0 = a1; 
  });
  // Title
  ctx.fillStyle = '#111827'; ctx.font = `${14*devicePixelRatio}px Inter, Arial`;
  ctx.textAlign='center'; ctx.fillText('Distribuição de Sentimentos', cx, 24*devicePixelRatio);
  
  // Labels on pie segments with percentages below
  let labelAngle = -Math.PI/2;
  segments.forEach((s,i)=>{
    const segmentAngle = 2*Math.PI*(s.value/total);
    const midAngle = labelAngle + segmentAngle/2;
    const labelRadius = r * 0.7; // Position labels inside the segments
    const labelX = cx + Math.cos(midAngle) * labelRadius;
    const labelY = cy + Math.sin(midAngle) * labelRadius;
    
    // Calculate percentage
    const percentage = Math.round((s.value / total) * 100);
    
    // Draw sentiment label
    ctx.fillStyle = '#ffffff'; // White text for visibility
    ctx.font = `bold ${12*devicePixelRatio}px Inter, Arial`;
    ctx.textAlign = 'center';
    ctx.fillText(s.label, labelX, labelY);
    
    // Draw percentage below the label
    ctx.font = `bold ${10*devicePixelRatio}px Inter, Arial`;
    ctx.fillText(`${percentage}%`, labelX, labelY + 16*devicePixelRatio);
    
    labelAngle += segmentAngle;
  })
}

function drawBar(canvasId, labels, values, title){
  console.log('drawBar called with:', canvasId, labels, values, title);
  const c = document.getElementById(canvasId);
  if (!c) {
    console.error('Canvas element not found:', canvasId);
    return;
  }
  const ctx = c.getContext('2d');
  const w = c.width = c.offsetWidth * devicePixelRatio;
  const h = c.height = c.offsetHeight * devicePixelRatio;
  ctx.clearRect(0,0,w,h);
  const margin = 36*devicePixelRatio; const bottom = 30*devicePixelRatio; const left = 40*devicePixelRatio;
  const cw = w - left - margin; const ch = h - bottom - margin;
  const max = Math.max(...values) * 1.15;
  
  // Professional color palette for emotions
  const getEmotionColor = (label) => {
    const colors = {
      'Raiva': '#dc2626',        // Red
      'Alegria': '#16a34a',      // Green  
      'Amor': '#e11d48',         // Rose
      'Medo': '#7c3aed',         // Purple
      'Tristeza': '#0ea5e9',     // Blue
      'Confiança': '#059669',    // Emerald
      'Ausente': '#6b7280'       // Gray
    };
    return colors[label] || '#6b7280';
  };
  
  // Axes
  ctx.strokeStyle = '#e5e7eb'; ctx.beginPath();
  ctx.moveTo(left, margin); ctx.lineTo(left, margin+ch); ctx.lineTo(left+cw, margin+ch); ctx.stroke();
  // Bars
  const bw = cw / (values.length*1.6);
  const total = values.reduce((sum, val) => sum + val, 0);
  
  values.forEach((v,i)=>{
    const x = left + (i+0.5)*bw*1.6;
    const y = margin + ch - (v/max)*ch;
    ctx.fillStyle = getEmotionColor(labels[i]); ctx.fillRect(x, y, bw, (v/max)*ch);
    
    // Calculate and draw percentage on top of bar
    const percentage = Math.round((v / total) * 100);
    ctx.fillStyle = '#374151';
    ctx.font = `bold ${10*devicePixelRatio}px Inter, Arial`;
    ctx.textAlign = 'center';
    ctx.fillText(`${percentage}%`, x + bw/2, y - 5*devicePixelRatio);
    
    // Label
    ctx.save(); ctx.fillStyle = '#374151'; ctx.font = `${12*devicePixelRatio}px Inter, Arial`; ctx.textAlign = 'center';
    ctx.translate(x + bw/2, margin+ch + 16*devicePixelRatio); ctx.rotate(-Math.PI/16);
    ctx.fillText(labels[i], 0, 0); ctx.restore();
  })
  // Title
  ctx.fillStyle = '#111827'; ctx.font = `${14*devicePixelRatio}px Inter, Arial`; ctx.textAlign='center';
  ctx.fillText(title, w/2, 22*devicePixelRatio);
}

function drawHBar(canvasId, labels, values, title){
  console.log('drawHBar called with:', canvasId, labels, values, title);
  const c = document.getElementById(canvasId);
  if (!c) {
    console.error('Canvas element not found:', canvasId);
    return;
  }
  const ctx = c.getContext('2d');
  const w = c.width = c.offsetWidth * devicePixelRatio;
  const h = c.height = c.offsetHeight * devicePixelRatio;
  ctx.clearRect(0,0,w,h);
  const margin = 80*devicePixelRatio; const right = 24*devicePixelRatio; const top = 42*devicePixelRatio;
  const cw = w - margin - right; const ch = h - margin - top;
  const max = Math.max(...values) * 1.1;
  const rowH = ch / values.length * .75; const gap = ch / values.length * .25;
  
  // Professional color palette for emotions (same as bar chart)
  const getEmotionColor = (label) => {
    const colors = {
      'Raiva': '#dc2626',        // Red
      'Alegria': '#16a34a',      // Green  
      'Amor': '#e11d48',         // Rose
      'Medo': '#7c3aed',         // Purple
      'Tristeza': '#0ea5e9',     // Blue
      'Confiança': '#059669',    // Emerald
      'Ausente': '#6b7280'       // Gray
    };
    return colors[label] || '#6b7280';
  };
  
  // Grid
  ctx.strokeStyle = '#eef2ff';
  for(let i=0;i<=5;i++){ const x = margin + (cw/5)*i; ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top+ch); ctx.stroke(); }
  // Rows
  values.forEach((v,i)=>{
    const y = top + i*(rowH+gap);
    ctx.fillStyle = getEmotionColor(labels[i]); ctx.fillRect(margin, y, (v/max)*cw, rowH);
    
    // Draw emotion label
    ctx.fillStyle = '#374151'; ctx.font = `${12*devicePixelRatio}px Inter, Arial`; ctx.textAlign='right';
    ctx.fillText(labels[i], margin - 16*devicePixelRatio, y + rowH*.7);
    
    // Draw count value at the end of the bar
    ctx.fillStyle = '#374151'; ctx.font = `bold ${11*devicePixelRatio}px Inter, Arial`; ctx.textAlign='left';
    ctx.fillText(v.toString(), margin + (v/max)*cw + 8*devicePixelRatio, y + rowH*.7);
  })
  // Axis labels
  ctx.fillStyle = '#374151'; ctx.font = `${12*devicePixelRatio}px Inter, Arial`; ctx.textAlign='center';
  ctx.fillText('Quantidade de Textos', margin + cw/2, top + ch + 26*devicePixelRatio);
  // Title
  ctx.fillStyle = '#111827'; ctx.font = `${14*devicePixelRatio}px Inter, Arial`;
  ctx.fillText(title, w/2, 20*devicePixelRatio);
}

function drawAllCharts(data = csvData){
  console.log('Starting to draw all charts...');
  if (downloadBtn) {
    downloadBtn.disabled = !data || data.length === 0;
  }
  
  if (data.length === 0) {
    console.warn('No data available, using dummy data');
    drawDummyCharts();
    return;
  }
  
  // Get selected model to determine which charts to show
  const selectedModel = modelSelect.value;
  
  // Process the data to get emotion and feeling counts
  const { emotionCounts, feelingCounts } = processEmotionData(data);
  console.log('Processed data:', { emotionCounts, feelingCounts });
  
  // For BERTimbau Emoções, only show emotion charts (no sentiment chart)
  if (selectedModel === 'bert-emotions') {
    // Hide sentiment chart (pie chart)
    const pieChart = document.getElementById('pie').parentElement;
    pieChart.style.display = 'none';
    
    // Show emotion charts
    modelInfo.style.display = 'none';
    document.getElementById('bar').style.display = 'block';
    document.getElementById('hbar').style.display = 'block';
    
    // Adjust grid layout for 2 charts - make them side by side
    const chartsGrid = document.querySelector('.charts-grid');
    chartsGrid.style.gridTemplateColumns = '1fr 1fr';
    chartsGrid.style.gridTemplateRows = 'auto';
    
    // Remove rank class from hbar to make it same size as bar
    const hbarBox = document.getElementById('hbar').parentElement;
    hbarBox.classList.remove('rank');
    hbarBox.style.height = '220px';
    
    // Prepare bar chart data (emotions)
    const emoLabels = Object.keys(emotionCounts);
    const emoValues = Object.values(emotionCounts);
    console.log('Drawing bar chart with data:', emoLabels, emoValues);
    drawBar('bar', emoLabels, emoValues, 'Distribuição de Emoções');

    // Sorted ranking (descending)
    const pairs = emoLabels.map((l,i)=>({l,v:emoValues[i]})).sort((a,b)=>b.v-a.v);
    console.log('Drawing horizontal bar chart with data:', pairs);
    drawHBar('hbar', pairs.map(p=>p.l), pairs.map(p=>p.v), 'Ranking das Emoções');
    
    console.log('BERTimbau Emoções - showing only emotion charts');
    return;
  }
  
  // Prepare pie chart data (sentiments) - for sentiment models
  const pieData = [
    { label: 'Negativo', value: feelingCounts.Negativo },
    { label: 'Positivo', value: feelingCounts.Positivo },
    { label: 'Neutro', value: feelingCounts.Neutro }
  ].filter(item => item.value > 0); // Only show sentiments that have data
  
  console.log('Drawing pie chart with data:', pieData);
  drawPie('pie', pieData);
  document.getElementById('pie').style.display = 'block';

  // For sentiment-only models (MNB and BERTimbau Sentimentos), only show pie chart
  if (selectedModel === 'mnb' || selectedModel === 'bert-sentiment') {
    // Show info message
    modelInfo.style.display = 'block';
    
    // Hide emotion charts for sentiment-only models
    const barChart = document.getElementById('bar').parentElement;
    const hbarChart = document.getElementById('hbar').parentElement;
    barChart.style.display = 'none';
    hbarChart.style.display = 'none';
    
    // Center the pie chart - adjust grid layout for single chart
    const chartsGrid = document.querySelector('.charts-grid');
    chartsGrid.style.gridTemplateColumns = '1fr';
    chartsGrid.style.gridTemplateRows = 'auto';
    chartsGrid.style.justifyItems = 'center';
    
    const pieChart = document.getElementById('pie').parentElement;
    pieChart.style.maxWidth = '400px';
    pieChart.style.margin = '0 auto';
    
    console.log(`${selectedModel} selected - hiding emotion charts, showing only sentiments`);
    return;
  }
  
  // Reset grid layout for models that show all charts
  const chartsGrid = document.querySelector('.charts-grid');
  chartsGrid.style.gridTemplateColumns = '320px 1fr';
  chartsGrid.style.gridTemplateRows = 'auto 1fr';
  chartsGrid.style.justifyItems = 'stretch';
  
  // Restore rank class and show all charts
  const pieChart = document.getElementById('pie').parentElement;
  pieChart.style.display = 'block';
  pieChart.style.maxWidth = 'none';
  
  const barChart = document.getElementById('bar').parentElement;
  barChart.style.display = 'block';
  
  const hbarChart = document.getElementById('hbar').parentElement;
  hbarChart.classList.add('rank');
  hbarChart.style.display = 'block';
  hbarChart.style.height = '260px';
  
  // Prepare bar chart data (emotions)
  const emoLabels = Object.keys(emotionCounts);
  const emoValues = Object.values(emotionCounts);
  console.log('Drawing bar chart with data:', emoLabels, emoValues);
  drawBar('bar', emoLabels, emoValues, 'Distribuição de Emoções');

  // Sorted ranking (descending)
  const pairs = emoLabels.map((l,i)=>({l,v:emoValues[i]})).sort((a,b)=>b.v-a.v);
  console.log('Drawing horizontal bar chart with data:', pairs);
  drawHBar('hbar', pairs.map(p=>p.l), pairs.map(p=>p.v), 'Ranking das Emoções');
  
  console.log('All charts drawn with real data!');
}

function drawDummyCharts(){
  // Get selected model to determine which charts to show
  const selectedModel = modelSelect.value;
  
  // For BERTimbau Emoções, only show emotion charts (no sentiment chart)
  if (selectedModel === 'bert-emotions') {
    // Hide sentiment chart (pie chart)
    const pieChart = document.getElementById('pie').parentElement;
    pieChart.style.display = 'none';
    
    // Show emotion charts
    modelInfo.style.display = 'none';
    document.getElementById('bar').style.display = 'block';
    document.getElementById('hbar').style.display = 'block';
    
    // Adjust grid layout for 2 charts - make them side by side
    const chartsGrid = document.querySelector('.charts-grid');
    chartsGrid.style.gridTemplateColumns = '1fr 1fr';
    chartsGrid.style.gridTemplateRows = 'auto';
    
    // Remove rank class from hbar to make it same size as bar
    const hbarBox = document.getElementById('hbar').parentElement;
    hbarBox.classList.remove('rank');
    hbarBox.style.height = '220px';

    const emoLabels = ['Raiva','Alegria','Amor','Medo','Tristeza','Confiança','Ausente'];
    const emoValues = [10, 18, 15, 8, 12, 7, 4];
    drawBar('bar', emoLabels, emoValues, 'Distribuição de Emoções');

    const pairs = emoLabels.map((l,i)=>({l,v:emoValues[i]})).sort((a,b)=>b.v-a.v);
    drawHBar('hbar', pairs.map(p=>p.l), pairs.map(p=>p.v), 'Ranking das Emoções');
    return;
  }
  
  // Fallback to original dummy data if no CSV data is available
  const pieData = [
    { label:'Negativo', value:60 },
    { label:'Positivo', value:40 },
  ];
  drawPie('pie', pieData);
  document.getElementById('pie').style.display = 'block';

  // For sentiment-only models (MNB and BERTimbau Sentimentos), only show pie chart
  if (selectedModel === 'mnb' || selectedModel === 'bert-sentiment') {
    // Show info message
    modelInfo.style.display = 'block';
    
    // Hide emotion charts for sentiment-only models
    const barChart = document.getElementById('bar').parentElement;
    const hbarChart = document.getElementById('hbar').parentElement;
    barChart.style.display = 'none';
    hbarChart.style.display = 'none';
    
    // Center the pie chart - adjust grid layout for single chart
    const chartsGrid = document.querySelector('.charts-grid');
    chartsGrid.style.gridTemplateColumns = '1fr';
    chartsGrid.style.gridTemplateRows = 'auto';
    chartsGrid.style.justifyItems = 'center';
    
    const pieChart = document.getElementById('pie').parentElement;
    pieChart.style.maxWidth = '400px';
    pieChart.style.margin = '0 auto';
    
    console.log(`${selectedModel} selected - hiding emotion charts in dummy mode`);
    return;
  }
  
  // Reset grid layout for models that show all charts
  const chartsGrid = document.querySelector('.charts-grid');
  chartsGrid.style.gridTemplateColumns = '320px 1fr';
  chartsGrid.style.gridTemplateRows = 'auto 1fr';
  chartsGrid.style.justifyItems = 'stretch';
  
  // Restore all charts
  const pieChart = document.getElementById('pie').parentElement;
  pieChart.style.display = 'block';
  pieChart.style.maxWidth = 'none';
  
  modelInfo.style.display = 'none';
  document.getElementById('bar').style.display = 'block';
  document.getElementById('hbar').style.display = 'block';
  
  const hbarBox = document.getElementById('hbar').parentElement;
  hbarBox.classList.add('rank');
  hbarBox.style.height = '260px';

  const emoLabels = ['Raiva','Alegria','Amor','Medo','Tristeza','Confiança','Ausente'];
  const emoValues = [10, 18, 15, 8, 12, 7, 4];
  drawBar('bar', emoLabels, emoValues, 'Distribuição de Emoções');

  const pairs = emoLabels.map((l,i)=>({l,v:emoValues[i]})).sort((a,b)=>b.v-a.v);
  drawHBar('hbar', pairs.map(p=>p.l), pairs.map(p=>p.v), 'Ranking das Emoções');
}

// Test function to validate CSV parsing with sample data
function testCSVParsing() {
  const sampleCSV = `text;emotion;feeling
"Test text 1";AUSENTE;NEUTRO
"Test text 2";"RAIVA;CONFIANÇA";NEGATIVO
"Test text 3";RAIVA;NEGATIVO
"Test text 4";AUSENTE;NEUTRO
"Test text 5";CONFIANÇA;POSITIVO
"Test text 6";AUSENTE;NEUTRO
"Test text 7";"TRISTEZA;RAIVA";NEGATIVO`;

  const parsedData = parseCSV(sampleCSV);
  console.log('Test parsed data:', parsedData);
  
  const { emotionCounts, feelingCounts } = processEmotionData(parsedData);
  console.log('Test emotion counts:', emotionCounts);
  console.log('Test feeling counts:', feelingCounts);
  
  // Expected results:
  // emotionCounts: { Ausente: 3, Raiva: 3, Confiança: 2, Tristeza: 1 }
  // feelingCounts: { Positivo: 1, Negativo: 3, Neutro: 3 }
  
  return { parsedData, emotionCounts, feelingCounts };
}

// Run test on page load for development
if (typeof window !== 'undefined') {
  console.log('Running CSV parsing test...');
  testCSVParsing();
}

// Render charts if someone lands directly on the results screen (dev mode)
if(document.getElementById('screen-results').classList.contains('active')) drawAllCharts();
