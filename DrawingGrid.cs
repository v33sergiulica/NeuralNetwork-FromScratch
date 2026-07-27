using UnityEngine;
using UnityEngine.UI;
using UnityEngine.InputSystem; // Adaugam namespace-ul New Input System
using TMPro;

public class DrawingGrid : MonoBehaviour
{
    public int resolution = 28;
    public NeuralNetworkInference neuralNet; // Drag & Drop in Inspector
    public TMP_Text predictionText;          // Drag & Drop TextMeshPro UI
    
    private double[] pixels;
    private Texture2D texture;
    private Image displayImage; // Image UI Component on this object

    void Start()
    {
        pixels = new double[resolution * resolution];
        texture = new Texture2D(resolution, resolution);
        texture.filterMode = FilterMode.Point; // Pixelated look
        
        displayImage = GetComponent<Image>();
        displayImage.sprite = Sprite.Create(texture, new Rect(0, 0, resolution, resolution), new Vector2(0.5f, 0.5f));
        
        ClearGrid();
    }

    void Update()
    {
        // New Input System pentru Mouse
        if (Mouse.current == null) return;

        // Desenare pe click stanga (tinut apasat)
        if (Mouse.current.leftButton.isPressed)
        {
            Vector2 mousePos = Mouse.current.position.ReadValue();
            Vector2 localPoint;
            
            RectTransformUtility.ScreenPointToLocalPointInRectangle(displayImage.rectTransform, mousePos, null, out localPoint);
            
            // Mapare coordonate (-width/2 ... width/2) la (0 ... resolution)
            float w = displayImage.rectTransform.rect.width;
            float h = displayImage.rectTransform.rect.height;
            
            int x = (int)((localPoint.x + w / 2) / w * resolution);
            int y = (int)((localPoint.y + h / 2) / h * resolution);

            if (x >= 0 && x < resolution && y >= 0 && y < resolution)
            {
                Draw(x, y);
            }
        }

        // Stergere pe click dreapta (o singura data)
        if (Mouse.current.rightButton.wasPressedThisFrame)
        {
            ClearGrid();
        }
    }

    // Brush Settings
    [Range(0.01f, 1f)] public float noiseAmount = 0.1f;
    [Range(0.1f, 1f)] public float brushIntensity = 0.5f;

    void Draw(int x, int y)
    {
        // 3x3 Soft Brush Kernel (mai aproape de MNIST)
        // Centrul e puternic, vecinii sunt gri
        DrawBrushPixel(x, y, 1.0f);     
        
        DrawBrushPixel(x+1, y, 0.7f);   
        DrawBrushPixel(x-1, y, 0.7f);
        DrawBrushPixel(x, y+1, 0.7f);
        DrawBrushPixel(x, y-1, 0.7f);

        DrawBrushPixel(x+1, y+1, 0.3f); 
        DrawBrushPixel(x-1, y-1, 0.3f);
        DrawBrushPixel(x-1, y+1, 0.3f);
        DrawBrushPixel(x+1, y-1, 0.3f);

        texture.Apply();
        
        // Predictie in timp real
        if (neuralNet && neuralNet.IsLoaded)
        {
            int prediction = neuralNet.Predict(pixels);
            if (predictionText) predictionText.text = "Prediction: " + prediction;
        }
    }

    void DrawBrushPixel(int x, int y, float factor)
    {
         float noise = UnityEngine.Random.Range(-noiseAmount, noiseAmount);
         SetPixel(x, y, (brushIntensity * factor) + noise);
    }

    void SetPixel(int x, int y, double intensity)
    {
        if (x < 0 || x >= resolution || y < 0 || y >= resolution) return;
        
        // Unity Texture: (0,0) jos-stanga, (27,27) sus-dreapta.
        // MNIST: (0,0) sus-stanga, (27,27) jos-dreapta.
        // Deci inversam Y-ul pentru retea.
        int mnist_y = resolution - 1 - y;
        int mnist_x = x;
        int index = mnist_y * resolution + mnist_x; 
        
        Color current = texture.GetPixel(x, y);
        float newVal = Mathf.Clamp01(current.r + (float)intensity);
        texture.SetPixel(x, y, new Color(newVal, newVal, newVal));
        
        pixels[index] = newVal; 
    }

    public void ClearGrid()
    {
        for (int i = 0; i < pixels.Length; i++) pixels[i] = 0;
        for (int x = 0; x < resolution; x++)
            for (int y = 0; y < resolution; y++)
                texture.SetPixel(x, y, Color.black);
        
        texture.Apply();
        if (predictionText) predictionText.text = "Draw a digit";
    }
}
