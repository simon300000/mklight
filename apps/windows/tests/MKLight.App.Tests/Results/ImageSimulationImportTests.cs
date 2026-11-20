using System.Text.Json.Nodes;
using MKLight.App.Views.Analysis.Results;
using MKLight.App.Views.Analysis.Results.Readers;
using MKLight.App.Views.Analysis.Settings;

namespace MKLight.App.Tests.Results;

public class ImageSimulationImportTests
{
    /// <summary>A BGRA picture, top row first, from a function of (x, y) giving (r, g, b) bytes.</summary>
    static byte[] Picture(int w, int h, Func<int, int, (byte R, byte G, byte B)> f)
    {
        var px = new byte[w * h * 4];
        for (var y = 0; y < h; y++)
            for (var x = 0; x < w; x++)
            {
                var (r, g, b) = f(x, y);
                var o = (y * w + x) * 4;
                (px[o], px[o + 1], px[o + 2], px[o + 3]) = (b, g, r, 255);
            }
        return px;
    }

    [Fact]
    public void SrgbCodesDecodeToLinearLight()
    {
        var img = ImageSourceImport.Linear(Picture(1, 1, (_, _) => (255, 128, 0)), 1, 1, false, 1, 1)!;
        Assert.Equal(3, img.Channels);
        Assert.Equal(1.0, img.Data[0], 12);
        Assert.Equal(Math.Pow((128 / 255.0 + 0.055) / 1.055, 2.4), img.Data[1], 12); // 0.2158
        Assert.Equal(0.0, img.Data[2], 12);
        // The linear segment near black.
        var dark = ImageSourceImport.Linear(Picture(1, 1, (_, _) => (10, 10, 10)), 1, 1, true, 1, 1)!;
        Assert.Equal(1, dark.Channels);
        Assert.Equal(10 / 255.0 / 12.92, dark.Data[0], 12);
    }

    [Fact]
    public void AveragesInLinearLightNotInCodeValues()
    {
        // A black and white checkerboard averages to half the light (code 188), not code 128.
        var img = ImageSourceImport.Linear(Picture(4, 4, (x, y) => (x + y) % 2 == 0 ? ((byte)255, (byte)255, (byte)255) : ((byte)0, (byte)0, (byte)0)), 4, 4, true, 2, 2)!;
        Assert.All(img.Data, v => Assert.Equal(0.5, v, 12));
    }

    [Fact]
    public void CropsTheLongerSideToTheFieldAspect()
    {
        Assert.Equal((25.0, 0.0, 50.0, 50.0), ImageSourceImport.Crop(100, 50, 1));
        Assert.Equal((0.0, 25.0, 50.0, 50.0), ImageSourceImport.Crop(50, 100, 1));
        Assert.Equal((0.0, 0.0, 100.0, 50.0), ImageSourceImport.Crop(100, 50, 2));
        // A wide picture (red left third, green middle, blue right) cropped square keeps the middle.
        var img = ImageSourceImport.Linear(Picture(300, 100, (x, _) => x < 100 ? ((byte)255, (byte)0, (byte)0) : x < 200 ? ((byte)0, (byte)255, (byte)0) : ((byte)0, (byte)0, (byte)255)), 300, 100, false, 10, 10, 1.0)!;
        Assert.All(Enumerable.Range(0, 100), p =>
        {
            Assert.Equal(0.0, img.Data[p * 3], 12);
            Assert.Equal(1.0, img.Data[p * 3 + 1], 12);
            Assert.Equal(0.0, img.Data[p * 3 + 2], 12);
        });
    }

    [Fact]
    public void RowZeroIsTheBottomOfThePicture()
    {
        // Top half white, bottom half black.
        var img = ImageSourceImport.Linear(Picture(2, 4, (_, y) => y < 2 ? ((byte)255, (byte)255, (byte)255) : ((byte)0, (byte)0, (byte)0)), 2, 4, true, 2, 2, 1.0 / 2)!;
        Assert.Equal([0.0, 0.0, 1.0, 1.0], img.Data);
    }

    [Fact]
    public void AreaAverageWeighsPartialPixels()
    {
        // 3 source pixels to 2: each output covers 1.5 pixels.
        var img = ImageSourceImport.Linear(Picture(3, 1, (x, _) => x == 0 ? ((byte)255, (byte)255, (byte)255) : ((byte)0, (byte)0, (byte)0)), 3, 1, true, 2, 1, 3.0)!;
        Assert.Equal(1 / 1.5, img.Data[0], 12);
        Assert.Equal(0.0, img.Data[1], 12);
    }

    [Fact]
    public void TargetSizeIsAtMost512APixelSide()
    {
        Assert.Equal((256, 256), ImageSourceImport.TargetSize(new JsonArray(256, 256)));
        Assert.Equal((512, 512), ImageSourceImport.TargetSize(new JsonArray(1024, 1024)));
        Assert.Equal((512, 256), ImageSourceImport.TargetSize(new JsonArray(2048, 1024)));
        Assert.Equal((64, 64), ImageSourceImport.TargetSize(null));
    }

    [Fact]
    public void FieldAspectComesFromTheResult()
    {
        Assert.Equal(2.0, ImageSourceImport.FieldAspect(TestData.Obj("""{"size": [200, 100], "pixel_size_mm": [0.01, 0.01]}""")));
        Assert.Equal(1.03, ImageSourceImport.FieldAspect(TestData.Obj("""{"size": [256, 256], "pixel_size_mm": [0.0103, 0.01]}""")) ?? 0, 9);
        Assert.Null(ImageSourceImport.FieldAspect(TestData.Obj("""{"size": [256, 256]}""")));
    }

    [Fact]
    public void TheSourceIsABufferAndTheFileIsKept()
    {
        var img = new LinearImage(2, 1, 3, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        var settings = TestData.Obj("""{"size": [2, 1], "monochrome": true, "source": {"kind": "pattern", "pattern": "grid"}}""");
        var s = ImageSourceImport.WithImage(settings, @"C:\pictures\scene.png", img, 1.5);
        Assert.Equal("""{"kind":"buffer","width":2,"height":1,"channels":3,"data":[0.1,0.2,0.3,0.4,0.5,0.6]}""", s["source"]!.ToJsonString());
        Assert.Equal(@"C:\pictures\scene.png", Json.Str(s[ImageSourceImport.FileKey]));
        Assert.Equal(1.5, Json.Number(s[ImageSourceImport.AspectKey]));
        Assert.Equal(true, Json.Bool(s["monochrome"])); // other settings kept
        Assert.Equal("grid", Json.Str(settings["source"]!["pattern"])); // the input is not changed
        Assert.Equal("scene.png", ImageSourceImport.FileName(s));
        // The engine reads the buffer back.
        Assert.Equal(img.Data, LinearImage.FromSource(s["source"])!.Data);

        // Saved without the samples; a pattern forgets the file.
        var saved = ImageSourceImport.Persistable(s);
        Assert.False(saved.ContainsKey("source"));
        Assert.False(saved.ContainsKey(ImageSourceImport.AspectKey));
        Assert.Equal(@"C:\pictures\scene.png", Json.Str(saved[ImageSourceImport.FileKey]));
        var p = ImageSourceImport.WithPattern(s, "siemens_star");
        Assert.False(p.ContainsKey(ImageSourceImport.FileKey));
        Assert.Equal("""{"kind":"pattern","pattern":"siemens_star"}""", p["source"]!.ToJsonString());
        Assert.Same(settings, ImageSourceImport.Persistable(settings));
    }

    [Fact]
    public void SamplesAreDerivedAgainForAnotherSizeOrAspect()
    {
        var img = new LinearImage(2, 2, 1, [0, 0, 0, 0]);
        var s = ImageSourceImport.WithImage(TestData.Obj("""{"size": [2, 2]}"""), "a.png", img, 1.0);
        Assert.False(ImageSourceImport.NeedsRefresh(s, 1.0));
        Assert.True(ImageSourceImport.NeedsRefresh(s, 1.2));
        Assert.True(ImageSourceImport.NeedsRefresh(ImageSimulationSettings.WithSize(s, 128), 1.0));
        Assert.True(ImageSourceImport.NeedsRefresh(ImageSourceImport.Persistable(s), 1.0));
        Assert.False(ImageSourceImport.NeedsRefresh(TestData.Obj("""{"size": [2, 2]}"""), 1.0)); // no file
        Assert.True(ImageSourceImport.IsImageFile(@"C:\x\Photo.JPG"));
        Assert.False(ImageSourceImport.IsImageFile("notes.txt"));
    }
}
