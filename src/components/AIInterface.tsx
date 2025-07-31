import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useToast } from "@/components/ui/use-toast";
import { Loader2, Brain, Image, FileText, Zap, Key, Trash2, Upload, Download, Eye, Settings } from 'lucide-react';
import { genesisAI } from '../ai.js';

export const AIInterface = () => {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [textInput, setTextInput] = useState('');
  const [textOutput, setTextOutput] = useState('');
  const [selectedProvider, setSelectedProvider] = useState('local');
  const [imageFile, setImageFile] = useState(null);
  const [imageAnalysis, setImageAnalysis] = useState(null);
  const [pageAnalysis, setPageAnalysis] = useState(null);
  const [apiKeys, setApiKeys] = useState({
    perplexity: '',
    openai: ''
  });
  const fileInputRef = useRef(null);
  const { toast } = useToast();

  useEffect(() => {
    updateStatus();
    
    const handleAIReady = () => {
      updateStatus();
      toast({
        title: "AI Engine Ready",
        description: "Genesis AI has been initialized successfully!",
      });
    };

    window.addEventListener('genesis-ai-ready', handleAIReady);
    return () => window.removeEventListener('genesis-ai-ready', handleAIReady);
  }, []);

  const updateStatus = () => {
    setStatus(genesisAI.getStatus());
  };

  const handleTextGeneration = async () => {
    if (!textInput.trim()) {
      toast({
        title: "Error",
        description: "Please enter some text to generate",
        variant: "destructive"
      });
      return;
    }

    setLoading(true);
    try {
      const result = await genesisAI.generateText(textInput, {
        provider: selectedProvider,
        maxTokens: 200,
        temperature: 0.7
      });
      setTextOutput(result.text);
      toast({
        title: "Text Generated",
        description: `Generated using ${result.provider} (${result.model})`,
      });
    } catch (error) {
      toast({
        title: "Generation Failed",
        description: error.message,
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setImageFile(file);
      setImageAnalysis(null);
    }
  };

  const analyzeImage = async () => {
    if (!imageFile) return;

    setLoading(true);
    try {
      const img = document.createElement('img');
      img.src = URL.createObjectURL(imageFile);
      await new Promise<void>(resolve => { img.onload = () => resolve(); });
      
      const analysis = await genesisAI.analyzeImage(img);
      setImageAnalysis(analysis);
      toast({
        title: "Image Analyzed",
        description: `Top prediction: ${analysis.topPrediction.label}`,
      });
    } catch (error) {
      toast({
        title: "Analysis Failed",
        description: error.message,
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  const removeBackground = async () => {
    if (!imageFile) return;

    setLoading(true);
    try {
      const img = document.createElement('img');
      img.src = URL.createObjectURL(imageFile);
      await new Promise<void>(resolve => { img.onload = () => resolve(); });
      
      const blob = await genesisAI.removeBackground(img);
      const url = URL.createObjectURL(blob);
      
      // Create download link
      const a = document.createElement('a');
      a.href = url;
      a.download = 'background-removed.png';
      a.click();
      
      toast({
        title: "Background Removed",
        description: "Image downloaded successfully!",
      });
    } catch (error) {
      toast({
        title: "Background Removal Failed",
        description: error.message,
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  const analyzeCurrentPage = async () => {
    setLoading(true);
    try {
      const analysis = await genesisAI.analyzePage();
      setPageAnalysis(analysis);
      toast({
        title: "Page Analyzed",
        description: "Complete analysis available in Page Intelligence tab",
      });
    } catch (error) {
      toast({
        title: "Analysis Failed",
        description: error.message,
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  const saveApiKey = (provider) => {
    genesisAI.setApiKey(provider, apiKeys[provider]);
    toast({
      title: "API Key Saved",
      description: `${provider} API key has been saved`,
    });
    updateStatus();
  };

  const removeApiKey = (provider) => {
    genesisAI.removeApiKey(provider);
    setApiKeys(prev => ({ ...prev, [provider]: '' }));
    toast({
      title: "API Key Removed",
      description: `${provider} API key has been removed`,
    });
    updateStatus();
  };

  return (
    <div className="w-full space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
            Genesis AI Engine
          </h1>
          <p className="text-muted-foreground">
            Advanced AI capabilities powered by cutting-edge models
          </p>
        </div>
        {status && (
          <div className="flex items-center gap-2">
            <Badge variant={status.initialized ? "default" : "secondary"}>
              {status.initialized ? "Ready" : "Loading"}
            </Badge>
            {status.memoryUsage && (
              <Badge variant="outline">
                RAM: {status.memoryUsage.used}MB
              </Badge>
            )}
          </div>
        )}
      </div>

      {status && !status.initialized && (
        <Alert>
          <Brain className="h-4 w-4" />
          <AlertDescription>
            AI models are still loading. Some features may not be available yet.
          </AlertDescription>
        </Alert>
      )}

      <Tabs defaultValue="generation" className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="generation" className="flex items-center gap-2">
            <Brain className="h-4 w-4" />
            Generation
          </TabsTrigger>
          <TabsTrigger value="vision" className="flex items-center gap-2">
            <Image className="h-4 w-4" />
            Vision
          </TabsTrigger>
          <TabsTrigger value="intelligence" className="flex items-center gap-2">
            <FileText className="h-4 w-4" />
            Intelligence
          </TabsTrigger>
          <TabsTrigger value="performance" className="flex items-center gap-2">
            <Zap className="h-4 w-4" />
            Performance
          </TabsTrigger>
          <TabsTrigger value="settings" className="flex items-center gap-2">
            <Settings className="h-4 w-4" />
            Settings
          </TabsTrigger>
        </TabsList>

        <TabsContent value="generation" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Text Generation</CardTitle>
              <CardDescription>
                Generate text using multiple AI providers and models
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-2">
                <Select value={selectedProvider} onValueChange={setSelectedProvider}>
                  <SelectTrigger className="w-48">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="local">Local (GPT-2)</SelectItem>
                    <SelectItem value="perplexity">Perplexity (Llama 3.1)</SelectItem>
                    <SelectItem value="openai">OpenAI (GPT-3.5)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <Textarea
                placeholder="Enter your prompt here..."
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                className="min-h-32"
              />
              
              <Button 
                onClick={handleTextGeneration} 
                disabled={loading || !textInput.trim()}
                className="w-full"
              >
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Brain className="mr-2 h-4 w-4" />
                    Generate Text
                  </>
                )}
              </Button>
              
              {textOutput && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Generated Text</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="whitespace-pre-wrap text-sm">{textOutput}</p>
                  </CardContent>
                </Card>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="vision" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Image Processing</CardTitle>
              <CardDescription>
                Analyze images and remove backgrounds using AI
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-6 text-center">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  ref={fileInputRef}
                  className="hidden"
                />
                <Button
                  variant="outline"
                  onClick={() => fileInputRef.current?.click()}
                  className="mb-4"
                >
                  <Upload className="mr-2 h-4 w-4" />
                  Upload Image
                </Button>
                {imageFile && (
                  <div className="space-y-2">
                    <p className="text-sm text-muted-foreground">
                      Selected: {imageFile.name}
                    </p>
                    <img
                      src={URL.createObjectURL(imageFile)}
                      alt="Preview"
                      className="max-w-full max-h-48 mx-auto rounded-lg"
                    />
                  </div>
                )}
              </div>
              
              {imageFile && (
                <div className="flex gap-2">
                  <Button onClick={analyzeImage} disabled={loading} className="flex-1">
                    {loading ? (
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    ) : (
                      <Eye className="mr-2 h-4 w-4" />
                    )}
                    Analyze Image
                  </Button>
                  <Button onClick={removeBackground} disabled={loading} className="flex-1">
                    {loading ? (
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    ) : (
                      <Download className="mr-2 h-4 w-4" />
                    )}
                    Remove Background
                  </Button>
                </div>
              )}
              
              {imageAnalysis && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Analysis Results</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {imageAnalysis.predictions.slice(0, 5).map((pred, idx) => (
                        <div key={idx} className="flex justify-between items-center">
                          <span className="text-sm">{pred.label}</span>
                          <Badge variant="outline">{pred.confidence}%</Badge>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="intelligence" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Page Intelligence</CardTitle>
              <CardDescription>
                Comprehensive AI analysis of the current webpage
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button onClick={analyzeCurrentPage} disabled={loading} className="w-full">
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <FileText className="mr-2 h-4 w-4" />
                    Analyze Current Page
                  </>
                )}
              </Button>
              
              {pageAnalysis && (
                <div className="space-y-4">
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm">Sentiment Analysis</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="flex items-center justify-between">
                        <span className="capitalize">{pageAnalysis.sentiment.sentiment}</span>
                        <Badge variant={pageAnalysis.sentiment.sentiment === 'positive' ? 'default' : 'secondary'}>
                          {pageAnalysis.sentiment.confidence}%
                        </Badge>
                      </div>
                    </CardContent>
                  </Card>
                  
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm">Readability</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span>Difficulty:</span>
                          <Badge>{pageAnalysis.readabilityScore.score}</Badge>
                        </div>
                        <div className="flex justify-between text-sm text-muted-foreground">
                          <span>Words per sentence:</span>
                          <span>{pageAnalysis.readabilityScore.avgWordsPerSentence}</span>
                        </div>
                        <div className="flex justify-between text-sm text-muted-foreground">
                          <span>Total words:</span>
                          <span>{pageAnalysis.readabilityScore.totalWords}</span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                  
                  {pageAnalysis.keyTopics && (
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-sm">Key Topics</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="flex flex-wrap gap-2">
                          {pageAnalysis.keyTopics.slice(0, 8).map((topic, idx) => (
                            <Badge key={idx} variant="outline">
                              {topic.word} ({topic.count})
                            </Badge>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="performance" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>AI Engine Status</CardTitle>
              <CardDescription>
                Monitor performance and loaded models
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {status && (
                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium mb-2">Capabilities</h4>
                    <div className="grid grid-cols-2 gap-2">
                      {Object.entries(status.capabilities).map(([cap, enabled]) => (
                        <div key={cap} className="flex items-center justify-between p-2 border rounded">
                          <span className="text-sm capitalize">{cap.replace(/([A-Z])/g, ' $1')}</span>
                          <Badge variant={enabled ? "default" : "secondary"}>
                            {enabled ? "Ready" : "Loading"}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-medium mb-2">Loaded Models</h4>
                    <div className="space-y-1">
                      {status.loadedModels.map(model => (
                        <Badge key={model} variant="outline" className="mr-2">
                          {model}
                        </Badge>
                      ))}
                    </div>
                  </div>
                  
                  {status.memoryUsage && (
                    <div>
                      <h4 className="font-medium mb-2">Memory Usage</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Used: {status.memoryUsage.used}MB</span>
                          <span>Limit: {status.memoryUsage.limit}MB</span>
                        </div>
                        <Progress 
                          value={(status.memoryUsage.used / status.memoryUsage.limit) * 100} 
                          className="h-2"
                        />
                      </div>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="settings" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>API Configuration</CardTitle>
              <CardDescription>
                Configure external AI service providers
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {['perplexity', 'openai'].map(provider => (
                <div key={provider} className="space-y-2">
                  <label className="text-sm font-medium capitalize">{provider} API Key</label>
                  <div className="flex gap-2">
                    <Input
                      type="password"
                      placeholder={`Enter ${provider} API key...`}
                      value={apiKeys[provider]}
                      onChange={(e) => setApiKeys(prev => ({ ...prev, [provider]: e.target.value }))}
                    />
                    <Button 
                      size="sm" 
                      onClick={() => saveApiKey(provider)}
                      disabled={!apiKeys[provider].trim()}
                    >
                      <Key className="h-4 w-4" />
                    </Button>
                    <Button 
                      size="sm" 
                      variant="destructive"
                      onClick={() => removeApiKey(provider)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                  {status?.apiKeys.includes(provider) && (
                    <p className="text-xs text-green-600">✓ API key configured</p>
                  )}
                </div>
              ))}
              
              <Alert>
                <AlertDescription>
                  API keys are stored locally in your browser. For production use, 
                  consider using Supabase Edge Functions for secure key management.
                </AlertDescription>
              </Alert>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};