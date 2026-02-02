const form = document.getElementById('predict-form');
const textarea = document.getElementById('text');
const resultDiv = document.getElementById('result');
const errorDiv = document.getElementById('error');
const submitBtn = document.getElementById('submit-btn');

function show(el, text) {
  el.textContent = text;
  el.classList.remove('hidden');
}

function hide(el) {
  el.classList.add('hidden');
  el.textContent = '';
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  hide(resultDiv);
  hide(errorDiv);

  const text = textarea.value.trim();
  if (!text) {
    show(errorDiv, 'Please enter a message.');
    return;
  }

  submitBtn.disabled = true;
  submitBtn.textContent = 'Predicting...';

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      throw new Error(data.detail || `Request failed with ${resp.status}`);
    }

    const data = await resp.json();
    const probPct = (data.probability * 100).toFixed(1);
    show(resultDiv, `Prediction: ${data.label.toUpperCase()} (${probPct}% confidence)`);
  } catch (err) {
    show(errorDiv, err.message || 'Something went wrong.');
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Predict';
  }
});
