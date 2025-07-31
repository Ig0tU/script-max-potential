/**
 * Genesis AI Engine - The Ultimate AI Powerhouse
 * Features: Multi-model inference, in-browser AI, image processing, intelligent automation
 */

import { pipeline, env } from '@huggingface/transformers';

// Configure transformers.js for optimal performance
env.allowLocalModels = false;
env.useBrowserCache = true;

export class GenesisAI {
  constructor() {
    this.models = new Map();
    this.isInitialized = false;
    this.capabilities = {
      textGeneration: false,
      imageClassification: false,
      speechRecognition: false,
      textEmbedding: false,
      imageSegmentation: false,
      sentimentAnalysis: false
    };
    this.apiKeys = {
      perplexity: localStorage.getItem('genesis_perplexity_key') || null,
      openai: localStorage.getItem('genesis_openai_key') || null
    };
  }

  async initialize() {
    if (this.isInitialized) return;
    
    console.log('🚀 Initializing Genesis AI Engine...');
    
    try {
      // Initialize core models in parallel for speed
      await Promise.allSettled([
        this.initTextGeneration(),
        this.initImageClassification(),
        this.initSentimentAnalysis(),
        this.initTextEmbedding()
      ]);
      
      this.isInitialized = true;
      console.log('✅ Genesis AI Engine initialized successfully');
      
      // Emit custom event for UI updates
      window.dispatchEvent(new CustomEvent('genesis-ai-ready', { 
        detail: { capabilities: this.capabilities } 
      }));
      
    } catch (error) {
      console.error('❌ Failed to initialize Genesis AI:', error);
      throw error;
    }
  }

  // Text Generation with multiple providers
  async initTextGeneration() {
    try {
      // Use lightweight model for in-browser generation
      this.models.set('textGen', await pipeline(
        'text-generation',
        'onnx-community/gpt2',
        { device: 'webgpu' }
      ));
      this.capabilities.textGeneration = true;
      console.log('✅ Text generation model loaded');
    } catch (error) {
      console.warn('⚠️ Text generation model failed to load:', error.message);
    }
  }

  async initImageClassification() {
    try {
      this.models.set('imageClassifier', await pipeline(
        'image-classification',
        'onnx-community/mobilenetv4_conv_small.e2400_r224_in1k',
        { device: 'webgpu' }
      ));
      this.capabilities.imageClassification = true;
      console.log('✅ Image classification model loaded');
    } catch (error) {
      console.warn('⚠️ Image classification model failed to load:', error.message);
    }
  }

  async initSentimentAnalysis() {
    try {
      this.models.set('sentiment', await pipeline(
        'sentiment-analysis',
        'onnx-community/distilbert-base-uncased-finetuned-sst-2-english',
        { device: 'webgpu' }
      ));
      this.capabilities.sentimentAnalysis = true;
      console.log('✅ Sentiment analysis model loaded');
    } catch (error) {
      console.warn('⚠️ Sentiment analysis model failed to load:', error.message);
    }
  }

  async initTextEmbedding() {
    try {
      this.models.set('embedding', await pipeline(
        'feature-extraction',
        'mixedbread-ai/mxbai-embed-xsmall-v1',
        { device: 'webgpu' }
      ));
      this.capabilities.textEmbedding = true;
      console.log('✅ Text embedding model loaded');
    } catch (error) {
      console.warn('⚠️ Text embedding model failed to load:', error.message);
    }
  }

  // Advanced Text Generation with multiple providers
  async generateText(prompt, options = {}) {
    const {
      provider = 'local',
      maxTokens = 100,
      temperature = 0.7,
      model = 'gpt2'
    } = options;

    try {
      if (provider === 'perplexity' && this.apiKeys.perplexity) {
        return await this.generateWithPerplexity(prompt, { maxTokens, temperature });
      } else if (provider === 'openai' && this.apiKeys.openai) {
        return await this.generateWithOpenAI(prompt, { maxTokens, temperature, model });
      } else {
        return await this.generateTextLocal(prompt, { maxTokens, temperature });
      }
    } catch (error) {
      console.error('Text generation failed:', error);
      throw new Error(`Text generation failed: ${error.message}`);
    }
  }

  async generateTextLocal(prompt, options = {}) {
    if (!this.capabilities.textGeneration) {
      throw new Error('Text generation model not available');
    }

    const model = this.models.get('textGen');
    const result = await model(prompt, {
      max_new_tokens: options.maxTokens,
      temperature: options.temperature,
      do_sample: true
    });

    return {
      text: result[0].generated_text,
      provider: 'local',
      model: 'gpt2'
    };
  }

  async generateWithPerplexity(prompt, options = {}) {
    const response = await fetch('https://api.perplexity.ai/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKeys.perplexity}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'llama-3.1-sonar-large-128k-online',
        messages: [
          { role: 'system', content: 'You are a helpful AI assistant. Be creative and informative.' },
          { role: 'user', content: prompt }
        ],
        temperature: options.temperature || 0.7,
        max_tokens: options.maxTokens || 1000,
      }),
    });

    if (!response.ok) {
      throw new Error(`Perplexity API error: ${response.statusText}`);
    }

    const data = await response.json();
    return {
      text: data.choices[0].message.content,
      provider: 'perplexity',
      model: 'llama-3.1-sonar-large-128k-online'
    };
  }

  async generateWithOpenAI(prompt, options = {}) {
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKeys.openai}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: options.model || 'gpt-3.5-turbo',
        messages: [
          { role: 'system', content: 'You are a helpful AI assistant.' },
          { role: 'user', content: prompt }
        ],
        temperature: options.temperature || 0.7,
        max_tokens: options.maxTokens || 1000,
      }),
    });

    if (!response.ok) {
      throw new Error(`OpenAI API error: ${response.statusText}`);
    }

    const data = await response.json();
    return {
      text: data.choices[0].message.content,
      provider: 'openai',
      model: options.model || 'gpt-3.5-turbo'
    };
  }

  // Image Analysis
  async analyzeImage(imageElement) {
    if (!this.capabilities.imageClassification) {
      throw new Error('Image classification model not available');
    }

    const model = this.models.get('imageClassifier');
    const results = await model(imageElement);
    
    return {
      predictions: results.map(r => ({
        label: r.label,
        confidence: Math.round(r.score * 100),
        score: r.score
      })),
      topPrediction: results[0]
    };
  }

  // Background Removal
  async removeBackground(imageElement) {
    try {
      console.log('Starting background removal...');
      
      // Initialize segmentation model if not already loaded
      if (!this.models.has('segmentation')) {
        this.models.set('segmentation', await pipeline(
          'image-segmentation',
          'Xenova/segformer-b0-finetuned-ade-512-512',
          { device: 'webgpu' }
        ));
      }

      const segmenter = this.models.get('segmentation');
      
      // Create canvas and resize if needed
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      const MAX_SIZE = 1024;
      let { width, height } = imageElement;
      
      if (width > MAX_SIZE || height > MAX_SIZE) {
        const ratio = Math.min(MAX_SIZE / width, MAX_SIZE / height);
        width *= ratio;
        height *= ratio;
      }
      
      canvas.width = width;
      canvas.height = height;
      ctx.drawImage(imageElement, 0, 0, width, height);
      
      // Process with segmentation
      const imageData = canvas.toDataURL('image/jpeg', 0.8);
      const result = await segmenter(imageData);
      
      if (!result || !result[0]?.mask) {
        throw new Error('Segmentation failed');
      }
      
      // Apply mask
      const outputCanvas = document.createElement('canvas');
      outputCanvas.width = width;
      outputCanvas.height = height;
      const outputCtx = outputCanvas.getContext('2d');
      
      outputCtx.drawImage(canvas, 0, 0);
      const imgData = outputCtx.getImageData(0, 0, width, height);
      
      // Apply inverted mask to keep subject
      for (let i = 0; i < result[0].mask.data.length; i++) {
        const alpha = Math.round((1 - result[0].mask.data[i]) * 255);
        imgData.data[i * 4 + 3] = alpha;
      }
      
      outputCtx.putImageData(imgData, 0, 0);
      
      return new Promise((resolve, reject) => {
        outputCanvas.toBlob(blob => {
          if (blob) resolve(blob);
          else reject(new Error('Failed to create blob'));
        }, 'image/png', 1.0);
      });
      
    } catch (error) {
      console.error('Background removal failed:', error);
      throw error;
    }
  }

  // Sentiment Analysis
  async analyzeSentiment(text) {
    if (!this.capabilities.sentimentAnalysis) {
      throw new Error('Sentiment analysis model not available');
    }

    const model = this.models.get('sentiment');
    const result = await model(text);
    
    return {
      sentiment: result[0].label.toLowerCase(),
      confidence: Math.round(result[0].score * 100),
      score: result[0].score
    };
  }

  // Text Embeddings
  async getEmbeddings(texts) {
    if (!this.capabilities.textEmbedding) {
      throw new Error('Text embedding model not available');
    }

    const model = this.models.get('embedding');
    const embeddings = await model(texts, { 
      pooling: 'mean', 
      normalize: true 
    });
    
    return embeddings.tolist();
  }

  // Semantic Search
  async semanticSearch(query, documents) {
    const queryEmbedding = await this.getEmbeddings([query]);
    const docEmbeddings = await this.getEmbeddings(documents);
    
    const similarities = docEmbeddings.map((docEmb, index) => ({
      document: documents[index],
      similarity: this.cosineSimilarity(queryEmbedding[0], docEmb),
      index
    }));
    
    return similarities
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, 10);
  }

  cosineSimilarity(a, b) {
    const dotProduct = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
    const magnitudeA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
    const magnitudeB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
    return dotProduct / (magnitudeA * magnitudeB);
  }

  // Smart Page Analysis
  async analyzePage() {
    const pageText = document.body.innerText;
    const images = Array.from(document.images);
    
    const analysis = {
      textSummary: await this.summarizeText(pageText.slice(0, 2000)),
      sentiment: await this.analyzeSentiment(pageText.slice(0, 500)),
      imageAnalysis: await Promise.all(
        images.slice(0, 5).map(async img => {
          try {
            return await this.analyzeImage(img);
          } catch (error) {
            return { error: error.message };
          }
        })
      ),
      keyTopics: await this.extractKeyTopics(pageText),
      readabilityScore: this.calculateReadability(pageText)
    };
    
    return analysis;
  }

  async summarizeText(text) {
    if (text.length < 100) return text;
    
    try {
      return await this.generateText(
        `Summarize this text in 2-3 sentences: ${text}`,
        { maxTokens: 100, provider: 'local' }
      );
    } catch (error) {
      return { error: 'Summarization failed', originalLength: text.length };
    }
  }

  async extractKeyTopics(text) {
    const words = text.toLowerCase()
      .replace(/[^\\w\\s]/g, '')
      .split(/\\s+/)
      .filter(word => word.length > 3);
    
    const frequency = {};
    words.forEach(word => {
      frequency[word] = (frequency[word] || 0) + 1;
    });
    
    return Object.entries(frequency)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 10)
      .map(([word, count]) => ({ word, count }));
  }

  calculateReadability(text) {
    const sentences = text.split(/[.!?]+/).length;
    const words = text.split(/\\s+/).length;
    const avgWordsPerSentence = words / sentences;
    
    let score = 'Unknown';
    if (avgWordsPerSentence < 15) score = 'Easy';
    else if (avgWordsPerSentence < 20) score = 'Medium';
    else score = 'Hard';
    
    return {
      score,
      avgWordsPerSentence: Math.round(avgWordsPerSentence),
      totalWords: words,
      totalSentences: sentences
    };
  }

  // API Key Management
  setApiKey(provider, key) {
    this.apiKeys[provider] = key;
    localStorage.setItem(`genesis_${provider}_key`, key);
    console.log(`✅ ${provider} API key updated`);
  }

  removeApiKey(provider) {
    this.apiKeys[provider] = null;
    localStorage.removeItem(`genesis_${provider}_key`);
    console.log(`🗑️ ${provider} API key removed`);
  }

  // Model Management
  async loadModel(type, modelId, options = {}) {
    try {
      console.log(`Loading ${type} model: ${modelId}...`);
      const model = await pipeline(type, modelId, { 
        device: 'webgpu',
        ...options 
      });
      this.models.set(modelId, model);
      console.log(`✅ Model ${modelId} loaded successfully`);
      return model;
    } catch (error) {
      console.error(`❌ Failed to load model ${modelId}:`, error);
      throw error;
    }
  }

  unloadModel(modelId) {
    if (this.models.has(modelId)) {
      this.models.delete(modelId);
      console.log(`🗑️ Model ${modelId} unloaded`);
    }
  }

  // Batch Processing
  async processBatch(items, processor, batchSize = 5) {
    const results = [];
    for (let i = 0; i < items.length; i += batchSize) {
      const batch = items.slice(i, i + batchSize);
      const batchResults = await Promise.allSettled(
        batch.map(item => processor(item))
      );
      results.push(...batchResults);
    }
    return results;
  }

  // Status and Diagnostics
  getStatus() {
    return {
      initialized: this.isInitialized,
      capabilities: this.capabilities,
      loadedModels: Array.from(this.models.keys()),
      apiKeys: Object.keys(this.apiKeys).filter(k => this.apiKeys[k]),
      memoryUsage: this.getMemoryUsage()
    };
  }

  getMemoryUsage() {
    if (performance.memory) {
      return {
        used: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024),
        total: Math.round(performance.memory.totalJSHeapSize / 1024 / 1024),
        limit: Math.round(performance.memory.jsHeapSizeLimit / 1024 / 1024)
      };
    }
    return null;
  }

  // Cleanup
  dispose() {
    this.models.clear();
    this.isInitialized = false;
    console.log('🧹 Genesis AI Engine disposed');
  }
}

// Create singleton instance
export const genesisAI = new GenesisAI();

// Auto-initialize when module loads
genesisAI.initialize().catch(console.error);

export default genesisAI;
