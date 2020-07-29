import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.seg.Segment;

/**
 * @Auther: Don
 * @Date: 2019/12/110:17
 * @Description:
 */
public class JavaHanlpSegmentCustom {
    private Segment SEGMENT;

    public JavaHanlpSegmentCustom() {
        init();
    }

    public void init() {
        System.setProperty("HANLP_ROOT", "./hanlp_dict_file"); //hanlp的词典包data文件夹的根目录
        SEGMENT = HanLP.newSegment().enableCustomDictionaryForcing(true); //强制优先适用自定义词典
    }
    public void main(String[] args){
        SEGMENT.seg("这就是陈奕迅");
    }
}