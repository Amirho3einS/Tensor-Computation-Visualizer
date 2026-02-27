import { GoogleGenAI } from "@google/genai";
import { ContractionSpec, DimensionSizes, VectorizationOption, OperationStep } from '../types';

export const explainContraction = async (
  contraction: ContractionSpec,
  sizes: DimensionSizes,
  vectorizeDim: string,
  option: VectorizationOption,
  vectorWidth: number,
  steps: OperationStep[]
): Promise<string> => {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

  // Build einsum string
  const einsumInputs = contraction.inputs.map(t => t.indices.join('')).join(',');
  const einsumOutput = contraction.output.indices.join('');
  const einsum = `${einsumInputs}->${einsumOutput}`;

  const prompt = `
    I am visualizing a tensor contraction for hardware with vector MAC units.
    
    Configuration:
    - Einsum: ${einsum}
    - Dimension sizes: ${JSON.stringify(sizes)}
    - Hardware vector width (w): ${vectorWidth}
    
    Vectorization Strategy:
    - Vectorizing along dimension '${vectorizeDim}'
    - Tensors being broadcast: ${option.broadcastTensors.length > 0 ? option.broadcastTensors.join(', ') : 'None (batch dimension)'}
    - Total MAC operations: ${steps.length}
    
    Explain briefly to a computer engineering student:
    1. Why this dimension choice requires broadcasting ${option.broadcastTensors.join(', ') || 'no tensors'}.
    2. The memory access pattern implications (contiguous vs. strided access).
    3. How this relates to the theorem that only output dimensions can be vectorized without horizontal reduction.
    
    Keep it concise (under 200 words). Use concrete examples from the current configuration.
  `;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: prompt,
    });
    return response.text || "No explanation generated.";
  } catch (error) {
    console.error("Gemini API Error:", error);
    return "Failed to generate explanation. Please check your API key.";
  }
};